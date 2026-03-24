"""
Feature Engineering: Convert raw election CSV data into ML-ready features
Aggregates zone-level data to province-level for consistent comparison across years
Includes both constituency (เขต) and partylist (บัญชีรายชื่อ) features
"""
import csv
import json
import os
import numpy as np
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_alignment():
    """Load party alignment mapping"""
    with open(os.path.join(DATA_DIR, "party_alignment.json"), encoding="utf-8") as f:
        return json.load(f)


def get_party_alignment(party_name, year, alignment_data):
    """Get alignment for a party in a given year"""
    year_str = str(year)
    for align_key, align_val in alignment_data.items():
        parties = align_val["parties"].get(year_str, [])
        if party_name in parties:
            return align_key
    return "others"


def load_constituency_data(year):
    """Load constituency CSV data for a given year"""
    fname = f"constituency_{year}_fixed.csv"
    fpath = os.path.join(DATA_DIR, fname)
    
    rows = []
    with open(fpath, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    print(f"  Loaded {len(rows)} rows from {fname}")
    return rows


def load_partylist_data(year):
    """Load partylist CSV data for a given year"""
    fname = f"partylist_{year}_fixed.csv"
    fpath = os.path.join(DATA_DIR, fname)
    
    rows = []
    with open(fpath, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    print(f"  Loaded {len(rows)} rows from {fname}")
    return rows


def aggregate_partylist_to_province(rows, year, alignment_data):
    """Aggregate partylist votes to province-level alignment shares"""
    provinces = defaultdict(lambda: {
        "total_votes": 0,
        "alignment_votes": defaultdict(int),
        "province_name": "",
        "region": "",
    })
    
    for row in rows:
        prov_id = row["province_id"]
        prov = provinces[prov_id]
        prov["province_name"] = row["province_name"]
        prov["region"] = row["region"]
        
        votes = int(row["votes"])
        party = row["party_name"]
        align = get_party_alignment(party, year, alignment_data)
        
        prov["total_votes"] += votes
        prov["alignment_votes"][align] += votes
    
    # Convert to shares
    result = {}
    for prov_id, prov in provinces.items():
        total = prov["total_votes"] if prov["total_votes"] > 0 else 1
        result[prov_id] = {
            "pl_progressive": round(prov["alignment_votes"].get("progressive", 0) / total * 100, 2),
            "pl_populist":    round(prov["alignment_votes"].get("populist", 0)    / total * 100, 2),
            "pl_conservative":round(prov["alignment_votes"].get("conservative", 0)/ total * 100, 2),
            "pl_others":      round(prov["alignment_votes"].get("others", 0)      / total * 100, 2),
        }
    return result


def aggregate_to_province(rows, year, alignment_data):
    """Aggregate zone-level data to province-level features"""
    provinces = defaultdict(lambda: {
        "zones": set(),
        "total_votes": 0,
        "total_eligible": 0,
        "total_invalid": 0,
        "party_votes": defaultdict(int),
        "alignment_votes": defaultdict(int),
        "alignment_seats": defaultdict(int),
        "winner_parties": [],
        "margins": [],
        "province_name": "",
        "region": "",
    })
    
    # Group by zone to find winners
    zone_data = defaultdict(list)
    for row in rows:
        prov_id = row["province_id"]
        zone_id = row["zone_id"]
        zone_data[(prov_id, zone_id)].append(row)
    
    for (prov_id, zone_id), zone_rows in zone_data.items():
        prov = provinces[prov_id]
        prov["zones"].add(zone_id)
        prov["province_name"] = zone_rows[0]["province_name"]
        prov["region"] = zone_rows[0]["region"]
        
        # Sort by votes descending
        sorted_rows = sorted(zone_rows, key=lambda x: int(x["votes"]), reverse=True)
        
        # Winner info
        winner = sorted_rows[0]
        winner_party = winner["party_name"]
        winner_votes = int(winner["votes"])
        prov["winner_parties"].append(winner_party)
        
        # Winner alignment
        winner_align = get_party_alignment(winner_party, year, alignment_data)
        prov["alignment_seats"][winner_align] += 1
        
        # Margin (winner - runner-up)
        if len(sorted_rows) > 1:
            runner_up_votes = int(sorted_rows[1]["votes"])
            prov["margins"].append(winner_votes - runner_up_votes)
        
        # Total votes per zone (from first row)
        total_votes_zone = int(winner.get("total_votes", 0))
        eligible_zone = int(winner.get("eligible_voters", 0))
        invalid_zone = int(winner.get("invalid_votes", 0))
        
        prov["total_votes"] += total_votes_zone if total_votes_zone > 0 else sum(int(r["votes"]) for r in zone_rows)
        prov["total_eligible"] += eligible_zone
        prov["total_invalid"] += invalid_zone
        
        # Party and alignment votes
        for row in zone_rows:
            party = row["party_name"]
            votes = int(row["votes"])
            prov["party_votes"][party] += votes
            
            align = get_party_alignment(party, year, alignment_data)
            prov["alignment_votes"][align] += votes
    
    return provinces


def compute_features(provinces, year):
    """Compute feature vector for each province"""
    features = {}
    
    for prov_id, prov in provinces.items():
        total_v = prov["total_votes"] if prov["total_votes"] > 0 else 1
        num_zones = len(prov["zones"])
        
        # Alignment shares
        prog_share = prov["alignment_votes"].get("progressive", 0) / total_v * 100
        pop_share = prov["alignment_votes"].get("populist", 0) / total_v * 100
        cons_share = prov["alignment_votes"].get("conservative", 0) / total_v * 100
        other_share = prov["alignment_votes"].get("others", 0) / total_v * 100
        
        # Top parties by votes
        sorted_parties = sorted(prov["party_votes"].items(), key=lambda x: x[1], reverse=True)
        top5_shares = []
        for i, (party, votes) in enumerate(sorted_parties[:5]):
            top5_shares.append(votes / total_v * 100)
        while len(top5_shares) < 5:
            top5_shares.append(0)
        
        # Dominant alignment
        align_seats = prov["alignment_seats"]
        dominant = max(align_seats, key=align_seats.get) if align_seats else "others"
        
        # Competitiveness (Herfindahl index)
        party_shares = [(v / total_v) for v in prov["party_votes"].values()]
        herfindahl = sum(s**2 for s in party_shares)
        
        # Turnout and invalid rates
        turnout = (prov["total_votes"] / prov["total_eligible"] * 100) if prov["total_eligible"] > 0 else 0
        invalid_rate = (prov["total_invalid"] / prov["total_votes"] * 100) if prov["total_votes"] > 0 else 0
        
        # Average margin
        avg_margin = np.mean(prov["margins"]) if prov["margins"] else 0
        avg_margin_pct = avg_margin / (total_v / num_zones) * 100 if num_zones > 0 else 0
        
        features[prov_id] = {
            "province_id": prov_id,
            "province_name": prov["province_name"],
            "region": prov["region"],
            "year": year,
            "num_zones": num_zones,
            "total_votes": prov["total_votes"],
            "total_eligible": prov["total_eligible"],
            "turnout_rate": round(turnout, 2),
            "invalid_rate": round(invalid_rate, 2),
            "progressive_share": round(prog_share, 2),
            "populist_share": round(pop_share, 2),
            "conservative_share": round(cons_share, 2),
            "other_share": round(other_share, 2),
            "top1_share": round(top5_shares[0], 2),
            "top2_share": round(top5_shares[1], 2),
            "top3_share": round(top5_shares[2], 2),
            "top4_share": round(top5_shares[3], 2),
            "top5_share": round(top5_shares[4], 2),
            "dominant_alignment": dominant,
            "progressive_seats": align_seats.get("progressive", 0),
            "populist_seats": align_seats.get("populist", 0),
            "conservative_seats": align_seats.get("conservative", 0),
            "avg_margin_pct": round(avg_margin_pct, 2),
            "herfindahl": round(herfindahl, 4),
            "competitiveness": round(1 - herfindahl, 4),
        }
    
    return features


def create_training_pairs(features_prev, features_curr, partylist_prev):
    """Create (input, target) pairs from two consecutive election years
    Including partylist features from previous year"""
    pairs = []
    
    # Match provinces that exist in both years
    common_provs = set(features_prev.keys()) & set(features_curr.keys())
    
    for prov_id in sorted(common_provs):
        prev = features_prev[prov_id]
        curr = features_curr[prov_id]
        pl = partylist_prev.get(prov_id, {"pl_progressive": 0, "pl_populist": 0, "pl_conservative": 0, "pl_others": 0})
        
        # Split-ticket diffs: constituency% - partylist%
        split_prog  = round(prev["progressive_share"] - pl["pl_progressive"], 2)
        split_pop   = round(prev["populist_share"]    - pl["pl_populist"],    2)
        split_cons  = round(prev["conservative_share"]- pl["pl_conservative"],2)
        
        # Region encoding (one-hot)
        regions = ["กรุงเทพมหานคร", "ภาคกลาง", "ภาคเหนือ", "ภาคตะวันออกเฉียงเหนือ",
                    "ภาคตะวันออก", "ภาคตะวันตก", "ภาคใต้"]
        region_onehot = [1 if prev["region"] == r else 0 for r in regions]
        
        # Input features (from previous year)
        input_feat = region_onehot + [
            prev["num_zones"],
            prev["turnout_rate"],
            prev["invalid_rate"],
            # Constituency alignment shares
            prev["progressive_share"],
            prev["populist_share"],
            prev["conservative_share"],
            prev["other_share"],
            # Top party shares
            prev["top1_share"],
            prev["top2_share"],
            prev["top3_share"],
            prev["top4_share"],
            prev["top5_share"],
            prev["avg_margin_pct"],
            prev["herfindahl"],
            prev["competitiveness"],
            # Partylist alignment shares
            pl["pl_progressive"],
            pl["pl_populist"],
            pl["pl_conservative"],
            pl["pl_others"],
            # Split-ticket diffs
            split_prog,
            split_pop,
            split_cons,
        ]
        
        pairs.append({
            "province_id": prov_id,
            "province_name": prev["province_name"],
            "region": prev["region"],
            "input": input_feat,
            "target_regression": [
                curr["progressive_share"],
                curr["populist_share"],
                curr["conservative_share"],
            ],
            "target_classification": curr["dominant_alignment"],
        })
    
    return pairs


if __name__ == "__main__":
    print("=" * 60)
    print("Feature Engineering (Constituency + Partylist)")
    print("=" * 60)
    
    alignment = load_alignment()
    
    # Load and process each year
    print("\n[1] Loading constituency data...")
    rows_62 = load_constituency_data(2562)
    rows_66 = load_constituency_data(2566)
    rows_69 = load_constituency_data(2569)
    
    print("\n[2] Loading partylist data...")
    pl_rows_62 = load_partylist_data(2562)
    pl_rows_66 = load_partylist_data(2566)
    pl_rows_69 = load_partylist_data(2569)
    
    print("\n[3] Aggregating constituency to province level...")
    prov_62 = aggregate_to_province(rows_62, 2562, alignment)
    prov_66 = aggregate_to_province(rows_66, 2566, alignment)
    prov_69 = aggregate_to_province(rows_69, 2569, alignment)
    print(f"  2562: {len(prov_62)} provinces")
    print(f"  2566: {len(prov_66)} provinces")
    print(f"  2569: {len(prov_69)} provinces")
    
    print("\n[4] Aggregating partylist to province level...")
    pl_62 = aggregate_partylist_to_province(pl_rows_62, 2562, alignment)
    pl_66 = aggregate_partylist_to_province(pl_rows_66, 2566, alignment)
    pl_69 = aggregate_partylist_to_province(pl_rows_69, 2569, alignment)
    
    print("\n[5] Computing constituency features...")
    feat_62 = compute_features(prov_62, 2562)
    feat_66 = compute_features(prov_66, 2566)
    feat_69 = compute_features(prov_69, 2569)
    
    print("\n[6] Creating training pairs (constituency + partylist)...")
    pairs_62_66 = create_training_pairs(feat_62, feat_66, pl_62)
    pairs_66_69 = create_training_pairs(feat_66, feat_69, pl_66)
    print(f"  2562→2566: {len(pairs_62_66)} pairs")
    print(f"  2566→2569: {len(pairs_66_69)} pairs")
    
    all_pairs = pairs_62_66 + pairs_66_69
    print(f"  Total training samples: {len(all_pairs)}")
    
    # Create prediction input (2569 → 2573)
    print("\n[7] Creating prediction input for 2573...")    
    pred_input = []
    regions = ["กรุงเทพมหานคร", "ภาคกลาง", "ภาคเหนือ", "ภาคตะวันออกเฉียงเหนือ",
                "ภาคตะวันออก", "ภาคตะวันตก", "ภาคใต้"]
    for prov_id, feat in sorted(feat_69.items()):
        region_onehot = [1 if feat["region"] == r else 0 for r in regions]
        pl = pl_69.get(prov_id, {"pl_progressive": 0, "pl_populist": 0, "pl_conservative": 0, "pl_others": 0})
        split_prog  = round(feat["progressive_share"] - pl["pl_progressive"], 2)
        split_pop   = round(feat["populist_share"]    - pl["pl_populist"],    2)
        split_cons  = round(feat["conservative_share"]- pl["pl_conservative"],2)
        input_feat = region_onehot + [
            feat["num_zones"], feat["turnout_rate"], feat["invalid_rate"],
            feat["progressive_share"], feat["populist_share"],
            feat["conservative_share"], feat["other_share"],
            feat["top1_share"], feat["top2_share"], feat["top3_share"],
            feat["top4_share"], feat["top5_share"],
            feat["avg_margin_pct"], feat["herfindahl"], feat["competitiveness"],
            # Partylist
            pl["pl_progressive"], pl["pl_populist"], pl["pl_conservative"], pl["pl_others"],
            # Split-ticket
            split_prog, split_pop, split_cons,
        ]
        pred_input.append({
            "province_id": prov_id,
            "province_name": feat["province_name"],
            "region": feat["region"],
            "input": input_feat,
        })
    print(f"  Prediction provinces: {len(pred_input)}")
    
    # Save all data
    output = {
        "feature_names": [
            "region_bkk", "region_central", "region_north", "region_northeast",
            "region_east", "region_west", "region_south",
            "num_zones", "turnout_rate", "invalid_rate",
            # Constituency alignment shares
            "progressive_share", "populist_share", "conservative_share", "other_share",
            "top1_share", "top2_share", "top3_share", "top4_share", "top5_share",
            "avg_margin_pct", "herfindahl", "competitiveness",
            # Partylist alignment shares
            "pl_progressive_share", "pl_populist_share", "pl_conservative_share", "pl_others_share",
            # Split-ticket (constituency - partylist)
            "split_progressive", "split_populist", "split_conservative",
        ],
        "alignment_labels": ["progressive", "populist", "conservative", "others"],
        "training_pairs": all_pairs,
        "prediction_input_2573": pred_input,
        "province_features": {
            "2562": feat_62,
            "2566": feat_66,
            "2569": feat_69,
        }
    }
    
    output_path = os.path.join(DATA_DIR, "ml_features.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved to {output_path} ({os.path.getsize(output_path):,} bytes)")
    
    # Summary
    print("\n" + "=" * 60)
    print("Feature Summary:")
    print(f"  Feature dimensions: {len(output['feature_names'])} features (was 22, now {len(output['feature_names'])})")
    print(f"    +4 partylist shares, +3 split-ticket diffs")
    print(f"  Training samples: {len(all_pairs)}")
    print(f"  Prediction samples: {len(pred_input)}")
    
    # Show alignment distribution
    print("\n  Alignment distribution (training targets):")
    from collections import Counter
    align_dist = Counter(p["target_classification"] for p in all_pairs)
    for a, c in align_dist.most_common():
        print(f"    {a:20s}: {c:3d} ({c/len(all_pairs)*100:.1f}%)")
    
    # Show sample
    print(f"\n  Sample pair (กรุงเทพฯ):")
    for p in all_pairs:
        if "pv_10" in p["province_id"]:
            print(f"    Province: {p['province_name']}")
            print(f"    Input ({len(p['input'])} features): {[round(x,2) for x in p['input'][:10]]}...")
            print(f"    Target regression: {p['target_regression']}")
            print(f"    Target classification: {p['target_classification']}")
            break
