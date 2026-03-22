"""
Fix party names: merge __NEXT_DATA__ party mapping with decoded Protobuf data
Then regenerate all CSV files with proper Thai party names + province names
"""
import requests
import json
import csv
import os
import re
from bs4 import BeautifulSoup

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def scrape_party_mapping():
    """Scrape party name mapping from __NEXT_DATA__ on The Standard"""
    print("[1] Fetching party mapping from The Standard...")
    resp = requests.get("https://election2569.thestandard.co/", timeout=30)
    resp.raise_for_status()
    resp.encoding = 'utf-8'  # Force UTF-8 encoding for Thai text
    
    soup = BeautifulSoup(resp.text, "html.parser")
    script = soup.find("script", id="__NEXT_DATA__")
    if not script:
        raise ValueError("__NEXT_DATA__ not found")
    
    data = json.loads(script.string)
    
    # Extract party data
    parties_raw = data["props"]["pageProps"]["initialState"]["masterStore"]["raw"]["parties"]
    
    party_map = {}  # id -> {partyName, partyCode, partyNo, partyColor, dataYear}
    for p in parties_raw:
        party_map[p["id"]] = {
            "partyName": p["partyName"],
            "partyCode": p["partyCode"],
            "partyNo": p.get("partyNo", 0),
            "dataYear": p["dataYear"],
        }
    
    # Also extract from rawParties for color info
    raw_parties = data["props"]["pageProps"]["initialState"]["masterStore"]["rawParties"]
    party_colors = {}
    for p in raw_parties:
        party_colors[p["id"]] = {
            "partyColor": p.get("partyColor", ""),
            "partyName": p.get("partyName", ""),
            "partyNameAbbrev": p.get("partyNameAbbrev", ""),
        }
    
    # Merge colors into party_map
    for pid, colors in party_colors.items():
        if pid in party_map:
            party_map[pid]["partyColor"] = colors["partyColor"]
            party_map[pid]["partyNameAbbrev"] = colors.get("partyNameAbbrev", "")
    
    # Extract regions
    regions = data["props"]["pageProps"]["initialState"]["masterStore"]["raw"]["regions"]
    region_map = {r["id"]: r["regionNameTH"] for r in regions}
    
    print(f"  Found {len(party_map)} parties")
    print(f"  Found {len(region_map)} regions")
    
    # Save party mapping
    with open(os.path.join(DATA_DIR, "party_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(party_map, f, ensure_ascii=False, indent=2)
    
    return party_map, region_map


def scrape_province_zone_mapping():
    """Scrape detailed province/zone mapping from The Standard's master.bin decoded data"""
    # Use the master.json we already decoded
    with open(os.path.join(DATA_DIR, "master.json"), encoding="utf-8") as f:
        master = json.load(f)
    
    zones = master.get("zones", {})
    
    # Build province map from zone IDs (pv_XX_z_Y -> province XX)
    province_ids = set()
    zone_to_province = {}
    for zid, z in zones.items():
        prov_id = z.get("provinceId", "")
        zone_to_province[zid] = prov_id
        if prov_id:
            province_ids.add(prov_id)
    
    return zone_to_province, province_ids


# Province code to Thai name mapping
PROVINCE_MAP = {
    "pv_10": "กรุงเทพมหานคร", "pv_11": "สมุทรปราการ", "pv_12": "นนทบุรี",
    "pv_13": "ปทุมธานี", "pv_14": "พระนครศรีอยุธยา", "pv_15": "อ่างทอง",
    "pv_16": "ลพบุรี", "pv_17": "สิงห์บุรี", "pv_18": "ชัยนาท",
    "pv_19": "สระบุรี", "pv_20": "ชลบุรี", "pv_21": "ระยอง",
    "pv_22": "จันทบุรี", "pv_23": "ตราด", "pv_24": "ฉะเชิงเทรา",
    "pv_25": "ปราจีนบุรี", "pv_26": "นครนายก", "pv_27": "สระแก้ว",
    "pv_30": "นครราชสีมา", "pv_31": "บุรีรัมย์", "pv_32": "สุรินทร์",
    "pv_33": "ศรีสะเกษ", "pv_34": "อุบลราชธานี", "pv_35": "ยโสธร",
    "pv_36": "ชัยภูมิ", "pv_37": "อำนาจเจริญ", "pv_38": "บึงกาฬ",
    "pv_39": "หนองบัวลำภู", "pv_40": "ขอนแก่น", "pv_41": "อุดรธานี",
    "pv_42": "เลย", "pv_43": "หนองคาย", "pv_44": "มหาสารคาม",
    "pv_45": "ร้อยเอ็ด", "pv_46": "กาฬสินธุ์", "pv_47": "สกลนคร",
    "pv_48": "นครพนม", "pv_49": "มุกดาหาร", "pv_50": "เชียงใหม่",
    "pv_51": "ลำพูน", "pv_52": "ลำปาง", "pv_53": "อุตรดิตถ์",
    "pv_54": "แพร่", "pv_55": "น่าน", "pv_56": "พะเยา",
    "pv_57": "เชียงราย", "pv_58": "แม่ฮ่องสอน", "pv_60": "นครสวรรค์",
    "pv_61": "อุทัยธานี", "pv_62": "กำแพงเพชร", "pv_63": "ตาก",
    "pv_64": "สุโขทัย", "pv_65": "พิษณุโลก", "pv_66": "พิจิตร",
    "pv_67": "เพชรบูรณ์", "pv_70": "ราชบุรี", "pv_71": "กาญจนบุรี",
    "pv_72": "สุพรรณบุรี", "pv_73": "นครปฐม", "pv_74": "สมุทรสาคร",
    "pv_75": "สมุทรสงคราม", "pv_76": "เพชรบุรี", "pv_77": "ประจวบคีรีขันธ์",
    "pv_80": "นครศรีธรรมราช", "pv_81": "กระบี่", "pv_82": "พังงา",
    "pv_83": "ภูเก็ต", "pv_84": "สุราษฎร์ธานี", "pv_85": "ระนอง",
    "pv_86": "ชุมพร", "pv_90": "สงขลา", "pv_91": "สตูล",
    "pv_92": "ตรัง", "pv_93": "พัทลุง", "pv_94": "ปัตตานี",
    "pv_95": "ยะลา", "pv_96": "นราธิวาส",
}

# Province to region mapping
PROVINCE_REGION = {
    "pv_10": "กรุงเทพมหานคร",
    **{f"pv_{i}": "ภาคกลาง" for i in [11,12,13,14,15,16,17,18,19,60,61,62,64,65,66,67]},
    **{f"pv_{i}": "ภาคตะวันออก" for i in [20,21,22,23,24,25,26,27]},
    **{f"pv_{i}": "ภาคตะวันออกเฉียงเหนือ" for i in [30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]},
    **{f"pv_{i}": "ภาคเหนือ" for i in [50,51,52,53,54,55,56,57,58,63]},
    **{f"pv_{i}": "ภาคตะวันตก" for i in [70,71,72,73,74,75,76,77]},
    **{f"pv_{i}": "ภาคใต้" for i in [80,81,82,83,84,85,86,90,91,92,93,94,95,96]},
}


def resolve_party_name(party_id_or_code, party_map, year):
    """Resolve party ID/code to Thai name"""
    # Direct lookup by full ID
    if party_id_or_code in party_map:
        return party_map[party_id_or_code]["partyName"]
    
    # Try with year prefix
    prefix = f"ely_y{year}_id_"
    full_id = f"{prefix}{party_id_or_code}"
    if full_id in party_map:
        return party_map[full_id]["partyName"]
    
    # Try numeric code lookup
    for pid, p in party_map.items():
        if str(p.get("partyCode", "")) == str(party_id_or_code):
            if str(year) in pid or p.get("dataYear", "") == f"y{year}":
                return p["partyName"]
    
    return str(party_id_or_code)


def fix_csv(input_path, output_path, party_map, year):
    """Fix CSV by resolving party names and province names"""
    rows = []
    with open(input_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            # Fix party name
            party_id = row.get("party_id", row.get("party_name", ""))
            row["party_name"] = resolve_party_name(party_id, party_map, year)
            
            # Fix province name
            prov_id = row.get("province_id", "")
            if prov_id and not row.get("province_name"):
                row["province_name"] = PROVINCE_MAP.get(prov_id, prov_id)
            elif not row.get("province_name"):
                # Extract from zone_id (pv_XX_z_Y)
                match = re.match(r'(pv_\d+)_z_\d+', row.get("zone_id", ""))
                if match:
                    prov_code = match.group(1)
                    row["province_name"] = PROVINCE_MAP.get(prov_code, prov_code)
                    row["province_id"] = prov_code
            
            # Fix region
            if not row.get("region") or row["region"] == "":
                prov_code = row.get("province_id", "")
                row["region"] = PROVINCE_REGION.get(prov_code, "")
            
            rows.append(row)
    
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"  -> Fixed {len(rows)} rows -> {output_path}")
    return rows


if __name__ == "__main__":
    print("=" * 60)
    print("Fixing Party Names & Province Names")
    print("=" * 60)
    
    # Step 1: Get party mapping from The Standard
    party_map, region_map = scrape_party_mapping()
    
    # Show some party names
    print("\n  Sample parties:")
    count = 0
    for pid, p in party_map.items():
        if count < 10:
            print(f"    {p['partyName']:25s} (code={p['partyCode']}, year={p['dataYear']})")
            count += 1
    
    # Step 2: Fix all CSV files
    print("\n[2] Fixing constituency_2569.csv...")
    fix_csv(
        os.path.join(DATA_DIR, "constituency_2569.csv"),
        os.path.join(DATA_DIR, "constituency_2569_fixed.csv"),
        party_map, 2569
    )
    
    print("\n[3] Fixing constituency_2566.csv...")
    fix_csv(
        os.path.join(DATA_DIR, "constituency_2566.csv"),
        os.path.join(DATA_DIR, "constituency_2566_fixed.csv"),
        party_map, 2566
    )
    
    print("\n[4] Fixing partylist_2569.csv...")
    fix_csv(
        os.path.join(DATA_DIR, "partylist_2569.csv"),
        os.path.join(DATA_DIR, "partylist_2569_fixed.csv"),
        party_map, 2569
    )
    
    print("\n[5] Fixing partylist_2566.csv...")
    fix_csv(
        os.path.join(DATA_DIR, "partylist_2566.csv"),
        os.path.join(DATA_DIR, "partylist_2566_fixed.csv"),
        party_map, 2566
    )
    
    # Step 3: Show sample fixed data
    print("\n" + "=" * 60)
    print("Sample Fixed Data (constituency_2569):")
    with open(os.path.join(DATA_DIR, "constituency_2569_fixed.csv"), encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < 8:
                print(f"  {row['province_name']:15s} เขต {row['zone_id'][-1]:>2s} | "
                      f"{row['party_name']:20s} | คะแนน {row['votes']:>7s} | อันดับ {row['rank']}")
    
    print("\n" + "=" * 60)
    print("DONE! Fixed files:")
    for f in sorted(os.listdir(DATA_DIR)):
        if "fixed" in f:
            fpath = os.path.join(DATA_DIR, f)
            size = os.path.getsize(fpath)
            print(f"  {f:45s} {size:>10,} bytes")
