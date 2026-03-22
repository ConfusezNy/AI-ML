"""
Convert decoded Protobuf election data -> structured CSV/JSON
Uses the decode_protobuf module to parse .bin files and outputs
clean, usable datasets.
"""
import struct
import json
import csv
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_raw")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# === Protobuf Decoder ===
def read_varint(data, pos):
    result = 0
    shift = 0
    while pos < len(data):
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return result, pos
        shift += 7
    raise ValueError("Unexpected end")


def decode_protobuf(data, depth=0):
    results = []
    pos = 0
    while pos < len(data):
        try:
            tag, pos = read_varint(data, pos)
            field_num = tag >> 3
            wire_type = tag & 0x07
            if field_num == 0 or field_num > 10000:
                break
            if wire_type == 0:
                value, pos = read_varint(data, pos)
                results.append((field_num, "varint", value))
            elif wire_type == 1:
                value = struct.unpack('<d', data[pos:pos+8])[0]
                pos += 8
                results.append((field_num, "double", value))
            elif wire_type == 2:
                length, pos = read_varint(data, pos)
                if length > len(data) - pos or length < 0:
                    break
                value = data[pos:pos+length]
                pos += length
                try:
                    str_val = value.decode('utf-8')
                    results.append((field_num, "string", str_val))
                except UnicodeDecodeError:
                    if depth < 5:
                        try:
                            nested = decode_protobuf(value, depth + 1)
                            if nested:
                                results.append((field_num, "message", nested))
                            else:
                                results.append((field_num, "bytes", value))
                        except:
                            results.append((field_num, "bytes", value))
                    else:
                        results.append((field_num, "bytes", value))
            elif wire_type == 5:
                value = struct.unpack('<f', data[pos:pos+4])[0]
                pos += 4
                results.append((field_num, "float", value))
            else:
                break
        except (IndexError, struct.error, ValueError):
            break
    return results


def get_field(fields, field_num, default=None):
    """Get first value of a field by number"""
    for fn, ft, fv in fields:
        if fn == field_num:
            return fv
    return default


def get_all_fields(fields, field_num):
    """Get all values of a field by number"""
    return [(ft, fv) for fn, ft, fv in fields if fn == field_num]


# === Master Data Parser ===
def parse_master(filepath):
    """Parse master.bin -> regions, provinces, zones, parties"""
    with open(filepath, "rb") as f:
        data = f.read()
    
    decoded = decode_protobuf(data)
    
    # Master has top-level fields:
    # 1 = regions, 2 = provinces, 3 = zones, 4 = parties, 5 = party scores
    regions = {}
    provinces = {}
    zones = {}
    parties = {}
    
    for field_num, field_type, field_val in decoded:
        if field_num == 1 and field_type == "message":
            # Region: field1=id, field2=nameTH
            rid = get_field(field_val, 1)
            rname = get_field(field_val, 2)
            if rid:
                regions[rid] = {"id": rid, "nameTH": rname}
        
        elif field_num == 2 and field_type == "message":
            # Province: field1=id, field2=nameTH, field3=regionId
            pid = get_field(field_val, 1)
            pname = get_field(field_val, 2)
            pregion = get_field(field_val, 3)
            if pid:
                provinces[pid] = {"id": pid, "nameTH": pname, "regionId": pregion}
        
        elif field_num == 3 and field_type == "message":
            # Zone: field1=id, field2=provinceId, field3=zoneNumber
            zid = get_field(field_val, 1)
            zprov = get_field(field_val, 2)
            znum = get_field(field_val, 3)
            if zid:
                zones[zid] = {"id": zid, "provinceId": zprov, "zoneNumber": znum}
        
        elif field_num == 4 and field_type == "message":
            # Party: field1=id, field2=partyCode, field3=partyName
            party_id = get_field(field_val, 1)
            party_code = get_field(field_val, 2)
            party_name = get_field(field_val, 3)
            if party_id:
                parties[party_id] = {
                    "id": party_id,
                    "partyCode": party_code,
                    "partyName": party_name,
                }
    
    return regions, provinces, zones, parties


# === Score Data Parser ===
def parse_score(filepath, master_parties):
    """Parse score-XX-ect.bin -> constituency results + party list results"""
    with open(filepath, "rb") as f:
        data = f.read()
    
    decoded = decode_protobuf(data)
    
    # Score file structure:
    # Field 1 = national summary (message)
    # Field 2 = zone results (repeated message)
    
    national_summary = None
    zone_results = []
    
    for field_num, field_type, field_val in decoded:
        if field_num == 1 and field_type == "message":
            national_summary = parse_national_summary(field_val, master_parties)
        elif field_num == 2 and field_type == "message":
            zone = parse_zone_result(field_val, master_parties)
            if zone:
                zone_results.append(zone)
    
    return national_summary, zone_results


def parse_national_summary(fields, master_parties):
    """Parse national summary from score data"""
    summary = {
        "total_eligible_voters": get_field(fields, 11),
        "reporting_status": get_field(fields, 12),
    }
    
    # Field 4 = constituency party seats
    const_parties = []
    for ft, fv in get_all_fields(fields, 4):
        if ft == "message":
            party_id = get_field(fv, 1)
            party_code = get_field(fv, 2)
            seats = get_field(fv, 3, 0)
            const_parties.append({
                "party_id": party_id,
                "party_code": party_code,
                "constituency_seats": seats,
                "party_name": master_parties.get(party_id, {}).get("partyName", ""),
            })
    
    # Field 5 = party list scores
    list_parties = []
    for ft, fv in get_all_fields(fields, 5):
        if ft == "message":
            party_id = get_field(fv, 1)
            party_code = get_field(fv, 2)
            votes = get_field(fv, 3, 0)
            rank = get_field(fv, 5, 0)
            list_parties.append({
                "party_id": party_id,
                "party_code": party_code,
                "party_list_votes": votes,
                "rank": rank,
                "party_name": master_parties.get(party_id, {}).get("partyName", ""),
            })
    
    summary["constituency_parties"] = const_parties
    summary["party_list_parties"] = list_parties
    return summary


def parse_zone_result(fields, master_parties):
    """Parse a single zone result"""
    zone_id = get_field(fields, 1)
    province_id = get_field(fields, 2)
    zone_number = get_field(fields, 3)
    
    if not zone_id:
        return None
    
    # Field 4 = constituency candidates in this zone
    candidates = []
    for ft, fv in get_all_fields(fields, 4):
        if ft == "message":
            party_id = get_field(fv, 1)
            party_code = get_field(fv, 2)
            candidate_id = get_field(fv, 3)
            votes = get_field(fv, 4, 0)
            vote_pct = get_field(fv, 5, 0)
            rank = get_field(fv, 6, 0)
            candidates.append({
                "party_id": party_id,
                "party_code": party_code,
                "candidate_id": candidate_id,
                "votes": votes,
                "vote_pct": vote_pct,
                "rank": rank,
                "party_name": master_parties.get(party_id, {}).get("partyName", ""),
            })
    
    # Field 5 = party list votes in this zone
    party_list = []
    for ft, fv in get_all_fields(fields, 5):
        if ft == "message":
            party_id = get_field(fv, 1)
            party_code = get_field(fv, 2)
            votes = get_field(fv, 3, 0)
            vote_pct = get_field(fv, 4, 0)
            rank = get_field(fv, 5, 0)
            party_list.append({
                "party_id": party_id,
                "party_code": party_code,
                "votes": votes,
                "vote_pct": vote_pct,
                "rank": rank,
                "party_name": master_parties.get(party_id, {}).get("partyName", ""),
            })
    
    # Summary fields
    eligible_voters = get_field(fields, 11)
    good_votes_const = get_field(fields, 6)
    total_votes_const = get_field(fields, 7)
    invalid_votes = get_field(fields, 8)
    no_vote = get_field(fields, 9)
    reporting_pct = get_field(fields, 10)
    
    return {
        "zone_id": zone_id,
        "province_id": province_id,
        "zone_number": zone_number,
        "eligible_voters": eligible_voters,
        "good_votes": good_votes_const,
        "total_votes": total_votes_const,
        "invalid_votes": invalid_votes,
        "no_vote": no_vote,
        "reporting_pct": reporting_pct,
        "constituency_candidates": candidates,
        "party_list_votes": party_list,
    }


# === CSV Writers ===
def write_constituency_csv(zone_results, provinces, year, output_path):
    """Write constituency results to CSV"""
    rows = []
    for zone in zone_results:
        prov = provinces.get(zone["province_id"], {})
        for cand in zone.get("constituency_candidates", []):
            rows.append({
                "year": year,
                "zone_id": zone["zone_id"],
                "province_id": zone["province_id"],
                "province_name": prov.get("nameTH", ""),
                "region": prov.get("regionId", ""),
                "zone_number": zone["zone_number"],
                "party_id": cand["party_id"],
                "party_name": cand["party_name"],
                "party_code": cand["party_code"],
                "candidate_id": cand.get("candidate_id", ""),
                "votes": cand["votes"],
                "vote_pct": cand["vote_pct"],
                "rank": cand["rank"],
                "eligible_voters": zone.get("eligible_voters", ""),
                "total_votes": zone.get("total_votes", ""),
                "invalid_votes": zone.get("invalid_votes", ""),
            })
    
    if rows:
        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  -> Wrote {len(rows)} rows to {output_path}")
    return rows


def write_partylist_csv(zone_results, provinces, year, output_path):
    """Write party list results to CSV"""
    rows = []
    for zone in zone_results:
        prov = provinces.get(zone["province_id"], {})
        for party in zone.get("party_list_votes", []):
            rows.append({
                "year": year,
                "zone_id": zone["zone_id"],
                "province_id": zone["province_id"],
                "province_name": prov.get("nameTH", ""),
                "region": prov.get("regionId", ""),
                "zone_number": zone["zone_number"],
                "party_id": party["party_id"],
                "party_name": party["party_name"],
                "party_code": party["party_code"],
                "votes": party["votes"],
                "vote_pct": party["vote_pct"],
                "rank": party["rank"],
            })
    
    if rows:
        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  -> Wrote {len(rows)} rows to {output_path}")
    return rows


def write_national_summary_csv(summary, year, output_path):
    """Write national party summary to CSV"""
    rows = []
    
    # Merge constituency seats + party list votes
    party_data = {}
    for p in summary.get("constituency_parties", []):
        pid = p["party_id"]
        party_data[pid] = {
            "year": year,
            "party_id": pid,
            "party_name": p["party_name"],
            "party_code": p["party_code"],
            "constituency_seats": p["constituency_seats"],
            "party_list_votes": 0,
            "party_list_rank": 0,
        }
    
    for p in summary.get("party_list_parties", []):
        pid = p["party_id"]
        if pid in party_data:
            party_data[pid]["party_list_votes"] = p["party_list_votes"]
            party_data[pid]["party_list_rank"] = p["rank"]
        else:
            party_data[pid] = {
                "year": year,
                "party_id": pid,
                "party_name": p["party_name"],
                "party_code": p["party_code"],
                "constituency_seats": 0,
                "party_list_votes": p["party_list_votes"],
                "party_list_rank": p["rank"],
            }
    
    rows = sorted(party_data.values(), key=lambda x: x["constituency_seats"], reverse=True)
    
    if rows:
        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  -> Wrote {len(rows)} rows to {output_path}")
    return rows


# === Main ===
if __name__ == "__main__":
    print("=" * 60)
    print("Election Data Converter: Protobuf -> CSV/JSON")
    print("=" * 60)
    
    # Step 1: Parse master data
    print("\n[1] Parsing master.bin...")
    master_path = os.path.join(DATA_DIR, "master.bin")
    regions, provinces, zones, parties = parse_master(master_path)
    print(f"  Regions: {len(regions)}")
    print(f"  Provinces: {len(provinces)}")
    print(f"  Zones: {len(zones)}")
    print(f"  Parties: {len(parties)}")
    
    # Save master data as JSON
    master_json = {
        "regions": regions,
        "provinces": provinces,
        "zones": zones,
        "parties": parties,
    }
    with open(os.path.join(OUTPUT_DIR, "master.json"), "w", encoding="utf-8") as f:
        json.dump(master_json, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved master.json")
    
    # Print sample data
    print(f"\n  Sample provinces:")
    for pid, prov in list(provinces.items())[:5]:
        print(f"    {pid}: {prov['nameTH']} ({prov['regionId']})")
    
    # Step 2: Parse score 2566
    print("\n[2] Parsing score-66-ect.bin (Election 2566)...")
    score66_path = os.path.join(DATA_DIR, "score_66.bin")
    if os.path.exists(score66_path):
        summary_66, zones_66 = parse_score(score66_path, parties)
        print(f"  Zone results: {len(zones_66)}")
        
        # Write CSVs
        write_constituency_csv(zones_66, provinces, 2566, 
                              os.path.join(OUTPUT_DIR, "constituency_2566.csv"))
        write_partylist_csv(zones_66, provinces, 2566,
                           os.path.join(OUTPUT_DIR, "partylist_2566.csv"))
        if summary_66:
            write_national_summary_csv(summary_66, 2566,
                                      os.path.join(OUTPUT_DIR, "national_summary_2566.csv"))
        
        # Save full JSON
        with open(os.path.join(OUTPUT_DIR, "score_2566.json"), "w", encoding="utf-8") as f:
            json.dump({"national_summary": summary_66, "zones": zones_66}, f, ensure_ascii=False, indent=2)
        print(f"  -> Saved score_2566.json")
    
    # Step 3: Parse score 2569
    print("\n[3] Parsing score-69-ect.bin (Election 2569)...")
    score69_path = os.path.join(DATA_DIR, "score_69.bin")
    if os.path.exists(score69_path):
        summary_69, zones_69 = parse_score(score69_path, parties)
        print(f"  Zone results: {len(zones_69)}")
        
        # Write CSVs
        write_constituency_csv(zones_69, provinces, 2569,
                              os.path.join(OUTPUT_DIR, "constituency_2569.csv"))
        write_partylist_csv(zones_69, provinces, 2569,
                           os.path.join(OUTPUT_DIR, "partylist_2569.csv"))
        if summary_69:
            write_national_summary_csv(summary_69, 2569,
                                      os.path.join(OUTPUT_DIR, "national_summary_2569.csv"))
        
        # Save full JSON
        with open(os.path.join(OUTPUT_DIR, "score_2569.json"), "w", encoding="utf-8") as f:
            json.dump({"national_summary": summary_69, "zones": zones_69}, f, ensure_ascii=False, indent=2)
        print(f"  -> Saved score_2569.json")
    
    # Step 4: Summary
    print("\n" + "=" * 60)
    print("DONE! Output files in data/ directory:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath)
        print(f"  {f:40s} {size:>10,} bytes")
    print("=" * 60)
