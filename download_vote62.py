"""
Download and convert 2562 election data from vote62.com
to match the same CSV format as 2566/2569 data
"""
import requests
import json
import csv
import os
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_DIR = os.path.join(os.path.dirname(__file__), "data_raw", "vote62")

URLS = {
    "ect100": "https://firebasestorage.googleapis.com/v0/b/thvote62.appspot.com/o/data%2Fect100.csv?alt=media",
    "candidate": "https://s3-ap-southeast-1.amazonaws.com/vote62/data/candidate.csv",
}

PROVINCE_MAP = {
    "กรุงเทพมหานคร": "pv_10", "สมุทรปราการ": "pv_11", "นนทบุรี": "pv_12",
    "ปทุมธานี": "pv_13", "พระนครศรีอยุธยา": "pv_14", "อ่างทอง": "pv_15",
    "ลพบุรี": "pv_16", "สิงห์บุรี": "pv_17", "ชัยนาท": "pv_18",
    "สระบุรี": "pv_19", "ชลบุรี": "pv_20", "ระยอง": "pv_21",
    "จันทบุรี": "pv_22", "ตราด": "pv_23", "ฉะเชิงเทรา": "pv_24",
    "ปราจีนบุรี": "pv_25", "นครนายก": "pv_26", "สระแก้ว": "pv_27",
    "นครราชสีมา": "pv_30", "บุรีรัมย์": "pv_31", "สุรินทร์": "pv_32",
    "ศรีสะเกษ": "pv_33", "อุบลราชธานี": "pv_34", "ยโสธร": "pv_35",
    "ชัยภูมิ": "pv_36", "อำนาจเจริญ": "pv_37", "บึงกาฬ": "pv_38",
    "หนองบัวลำภู": "pv_39", "ขอนแก่น": "pv_40", "อุดรธานี": "pv_41",
    "เลย": "pv_42", "หนองคาย": "pv_43", "มหาสารคาม": "pv_44",
    "ร้อยเอ็ด": "pv_45", "กาฬสินธุ์": "pv_46", "สกลนคร": "pv_47",
    "นครพนม": "pv_48", "มุกดาหาร": "pv_49", "เชียงใหม่": "pv_50",
    "ลำพูน": "pv_51", "ลำปาง": "pv_52", "อุตรดิตถ์": "pv_53",
    "แพร่": "pv_54", "น่าน": "pv_55", "พะเยา": "pv_56",
    "เชียงราย": "pv_57", "แม่ฮ่องสอน": "pv_58", "นครสวรรค์": "pv_60",
    "อุทัยธานี": "pv_61", "กำแพงเพชร": "pv_62", "ตาก": "pv_63",
    "สุโขทัย": "pv_64", "พิษณุโลก": "pv_65", "พิจิตร": "pv_66",
    "เพชรบูรณ์": "pv_67", "ราชบุรี": "pv_70", "กาญจนบุรี": "pv_71",
    "สุพรรณบุรี": "pv_72", "นครปฐม": "pv_73", "สมุทรสาคร": "pv_74",
    "สมุทรสงคราม": "pv_75", "เพชรบุรี": "pv_76", "ประจวบคีรีขันธ์": "pv_77",
    "นครศรีธรรมราช": "pv_80", "กระบี่": "pv_81", "พังงา": "pv_82",
    "ภูเก็ต": "pv_83", "สุราษฎร์ธานี": "pv_84", "ระนอง": "pv_85",
    "ชุมพร": "pv_86", "สงขลา": "pv_90", "สตูล": "pv_91",
    "ตรัง": "pv_92", "พัทลุง": "pv_93", "ปัตตานี": "pv_94",
    "ยะลา": "pv_95", "นราธิวาส": "pv_96",
}

PROVINCE_REGION = {
    "pv_10": "กรุงเทพมหานคร",
    **{f"pv_{i}": "ภาคกลาง" for i in [11,12,13,14,15,16,17,18,19,60,61,62,64,65,66,67]},
    **{f"pv_{i}": "ภาคตะวันออก" for i in [20,21,22,23,24,25,26,27]},
    **{f"pv_{i}": "ภาคตะวันออกเฉียงเหนือ" for i in [30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]},
    **{f"pv_{i}": "ภาคเหนือ" for i in [50,51,52,53,54,55,56,57,58,63]},
    **{f"pv_{i}": "ภาคตะวันตก" for i in [70,71,72,73,74,75,76,77]},
    **{f"pv_{i}": "ภาคใต้" for i in [80,81,82,83,84,85,86,90,91,92,93,94,95,96]},
}


def download_file(name, url, ext="csv"):
    """Download a single file with retry"""
    os.makedirs(RAW_DIR, exist_ok=True)
    fpath = os.path.join(RAW_DIR, f"{name}.{ext}")
    
    if os.path.exists(fpath) and os.path.getsize(fpath) > 100:
        print(f"  {name} already exists, skipping download")
        return fpath
    
    print(f"  Downloading {name}...")
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            resp.encoding = 'utf-8'
            
            with open(fpath, 'w', encoding='utf-8-sig', newline='') as f:
                f.write(resp.text.replace('\r\r\n', '\n').replace('\r\n', '\n'))
            
            print(f"    -> Saved ({len(resp.content):,} bytes)")
            return fpath
        except Exception as e:
            print(f"    -> Attempt {attempt+1} failed: {e}")
    
    return None


def build_candidate_map(cand_path):
    """Build mapping: (province, zone, no) -> party_name"""
    cand_map = {}
    with open(cand_path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['province'].strip(), row['zone'].strip(), row['no'].strip())
            cand_map[key] = row['party'].strip()
    
    print(f"  Built candidate map: {len(cand_map)} entries")
    return cand_map


def convert_to_constituency_csv(ect_path, cand_map, output_path):
    """Convert ect100.csv + candidate map to constituency CSV matching 2566/2569 format"""
    
    # Read ect100 data
    ect_rows = []
    with open(ect_path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ect_rows.append(row)
    
    print(f"  Read {len(ect_rows)} score rows from ect100.csv")
    
    # Group by (province, zone) and compute totals + ranks
    from collections import defaultdict
    zone_data = defaultdict(list)
    
    for row in ect_rows:
        province = row['province'].strip()
        zone = row['zone'].strip()
        no = row['no'].strip()
        score = int(row['score'].strip())
        
        # Lookup party name
        key = (province, zone, no)
        party_name = cand_map.get(key, f"unknown_{no}")
        
        zone_data[(province, zone)].append({
            'candidate_no': no,
            'party_name': party_name,
            'votes': score,
        })
    
    print(f"  Found {len(zone_data)} zones")
    
    # Build output rows
    output_rows = []
    for (province, zone), candidates in zone_data.items():
        # Sort by votes descending for ranking
        candidates.sort(key=lambda x: x['votes'], reverse=True)
        
        # Compute total votes in this zone
        total_votes = sum(c['votes'] for c in candidates)
        
        # Get province code
        prov_id = PROVINCE_MAP.get(province, "")
        region = PROVINCE_REGION.get(prov_id, "")
        zone_id = f"{prov_id}_z_{zone}" if prov_id else f"unknown_z_{zone}"
        
        for rank, cand in enumerate(candidates, 1):
            votes = cand['votes']
            vote_pct = (votes / total_votes * 100) if total_votes > 0 else 0
            
            output_rows.append({
                'year': '2562',
                'zone_id': zone_id,
                'province_id': prov_id,
                'province_name': province,
                'region': region,
                'zone_number': zone,
                'party_id': f"v62_{cand['candidate_no']}",
                'party_name': cand['party_name'],
                'party_code': cand['candidate_no'],
                'candidate_id': f"v62_cand_{cand['candidate_no']}",
                'votes': str(votes),
                'vote_pct': f"{vote_pct:.2f}",
                'rank': str(rank),
                'eligible_voters': '0',  # Not available in vote62 data
                'total_votes': str(total_votes),
                'invalid_votes': '0',     # Not available
            })
    
    # Write CSV
    fieldnames = ['year', 'zone_id', 'province_id', 'province_name', 'region',
                  'zone_number', 'party_id', 'party_name', 'party_code',
                  'candidate_id', 'votes', 'vote_pct', 'rank',
                  'eligible_voters', 'total_votes', 'invalid_votes']
    
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    
    print(f"  -> Wrote {len(output_rows)} rows to {output_path}")
    return output_rows


if __name__ == "__main__":
    print("=" * 60)
    print("2562 Election Data (vote62.com)")
    print("=" * 60)
    
    # Step 1: Download files
    print("\n[1] Downloading files...")
    ect_path = download_file("ect100", URLS["ect100"])
    cand_path = download_file("candidate", URLS["candidate"])
    
    if not ect_path or not cand_path:
        print("ERROR: Failed to download required files!")
        exit(1)
    
    # Step 2: Build candidate mapping
    print("\n[2] Building candidate -> party mapping...")
    cand_map = build_candidate_map(cand_path)
    
    # Show some parties
    parties = set(v for v in cand_map.values())
    print(f"  Found {len(parties)} unique parties")
    major = ['เพื่อไทย', 'ประชาธิปัตย์', 'พลังประชารัฐ', 'อนาคตใหม่', 'ภูมิใจไทย']
    for p in major:
        if p in parties:
            print(f"    ✅ {p}")
        else:
            print(f"    ❌ {p} (not found)")
    
    # Step 3: Convert to constituency CSV
    print("\n[3] Converting to constituency CSV...")
    output_path = os.path.join(DATA_DIR, "constituency_2562_fixed.csv")
    rows = convert_to_constituency_csv(ect_path, cand_map, output_path)
    
    # Step 4: Show summary
    print("\n" + "=" * 60)
    print("Summary:")
    from collections import Counter
    winners = [r['party_name'] for r in rows if r['rank'] == '1']
    print(f"\nTop parties (by constituency wins):")
    for party, count in Counter(winners).most_common(15):
        print(f"  {party:30s} {count:3d} เขต")
    
    zones = set(r['zone_id'] for r in rows)
    provinces = set(r['province_name'] for r in rows)
    print(f"\nTotal zones: {len(zones)}")
    print(f"Total provinces: {len(provinces)}")
    print(f"Total rows: {len(rows)}")
    
    # Sample data
    print(f"\nSample data:")
    for r in rows[:8]:
        print(f"  {r['province_name']:15s} เขต {r['zone_number']:>2s} | "
              f"{r['party_name']:20s} | คะแนน {r['votes']:>7s} | อันดับ {r['rank']}")
    
    print(f"\n✅ Output: {output_path}")
    print(f"   Size: {os.path.getsize(output_path):,} bytes")
