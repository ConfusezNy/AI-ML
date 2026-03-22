"""
Download and decode Protobuf .bin files from The Standard Election 2569
"""
import requests
import struct
import json
import os

# URLs
URLS = {
    "master": "https://election2569-assets.thestandard.co/data/master.bin",
    "score_69": "https://election2569-assets.thestandard.co/data/score-69-ect.bin",
    "score_66": "https://election2569-assets.thestandard.co/data/score-66-ect.bin",
    "config": "https://election2569-assets.thestandard.co/config/main-config.json",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_raw")
os.makedirs(DATA_DIR, exist_ok=True)


def download_file(name, url):
    """Download a file from URL"""
    filepath = os.path.join(DATA_DIR, f"{name}{'.json' if url.endswith('.json') else '.bin'}")
    if os.path.exists(filepath):
        print(f"[SKIP] {name} already downloaded: {filepath}")
        return filepath
    
    print(f"[DOWNLOAD] {name} from {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    
    with open(filepath, "wb") as f:
        f.write(resp.content)
    
    print(f"  -> Saved {len(resp.content):,} bytes to {filepath}")
    return filepath


def analyze_binary(filepath, name):
    """Analyze binary file structure"""
    with open(filepath, "rb") as f:
        data = f.read()
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"File size: {len(data):,} bytes")
    print(f"First 100 bytes (hex): {data[:100].hex()}")
    print(f"First 100 bytes (raw): {data[:100]}")
    
    # Check if it's gzip compressed
    if data[:2] == b'\x1f\x8b':
        print("  -> GZIP compressed!")
        import gzip
        data = gzip.decompress(data)
        print(f"  -> Decompressed size: {len(data):,} bytes")
        print(f"  -> First 100 bytes (hex): {data[:100].hex()}")
        print(f"  -> First 100 bytes (raw): {data[:100]}")
        
        # Save decompressed
        decomp_path = filepath + ".decompressed"
        with open(decomp_path, "wb") as f:
            f.write(data)
        print(f"  -> Saved decompressed to {decomp_path}")
    
    # Check if it's JSON
    try:
        text = data.decode('utf-8')
        parsed = json.loads(text)
        print("  -> It's JSON!")
        json_path = filepath + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        print(f"  -> Saved JSON to {json_path}")
        if isinstance(parsed, dict):
            print(f"  -> Top-level keys: {list(parsed.keys())[:20]}")
        elif isinstance(parsed, list):
            print(f"  -> List with {len(parsed)} items")
            if len(parsed) > 0:
                print(f"  -> First item: {json.dumps(parsed[0], ensure_ascii=False)[:200]}")
        return parsed
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    
    # Try to decode as protobuf (raw wire format)
    print("\n  Attempting raw protobuf decode...")
    try:
        decoded = decode_protobuf_raw(data)
        print(f"  -> Decoded {len(decoded)} fields")
        for i, (field_num, wire_type, value) in enumerate(decoded[:20]):
            val_preview = str(value)[:100] if isinstance(value, (str, bytes)) else value
            print(f"     Field {field_num} (type {wire_type}): {val_preview}")
        return decoded
    except Exception as e:
        print(f"  -> Raw decode failed: {e}")
    
    # Try MessagePack
    try:
        import msgpack
        parsed = msgpack.unpackb(data, raw=False)
        print("  -> It's MessagePack!")
        return parsed
    except Exception:
        pass
    
    return data


def decode_protobuf_raw(data, depth=0):
    """Decode raw protobuf wire format without .proto file"""
    results = []
    pos = 0
    max_fields = 500  # Limit to prevent infinite loops
    
    while pos < len(data) and len(results) < max_fields:
        try:
            # Read varint tag
            tag, pos = read_varint(data, pos)
            field_number = tag >> 3
            wire_type = tag & 0x07
            
            if field_number == 0 or field_number > 10000:
                break
            
            if wire_type == 0:  # Varint
                value, pos = read_varint(data, pos)
                results.append((field_number, wire_type, value))
            elif wire_type == 1:  # 64-bit
                value = struct.unpack('<d', data[pos:pos+8])[0]
                pos += 8
                results.append((field_number, wire_type, value))
            elif wire_type == 2:  # Length-delimited (string, bytes, or embedded message)
                length, pos = read_varint(data, pos)
                if length > len(data) - pos or length < 0:
                    break
                value = data[pos:pos+length]
                pos += length
                
                # Try to decode as UTF-8 string
                try:
                    str_value = value.decode('utf-8')
                    results.append((field_number, wire_type, str_value))
                except UnicodeDecodeError:
                    # Try as nested message
                    if depth < 3:
                        try:
                            nested = decode_protobuf_raw(value, depth + 1)
                            if len(nested) > 0:
                                results.append((field_number, "nested", nested))
                            else:
                                results.append((field_number, wire_type, value.hex()[:50]))
                        except:
                            results.append((field_number, wire_type, value.hex()[:50]))
                    else:
                        results.append((field_number, wire_type, value.hex()[:50]))
            elif wire_type == 5:  # 32-bit
                value = struct.unpack('<f', data[pos:pos+4])[0]
                pos += 4
                results.append((field_number, wire_type, value))
            else:
                break
        except (IndexError, struct.error):
            break
    
    return results


def read_varint(data, pos):
    """Read a protobuf varint"""
    result = 0
    shift = 0
    while pos < len(data):
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("Varint too long")
    raise ValueError("Unexpected end of data")


def protobuf_to_readable(decoded, indent=0):
    """Convert decoded protobuf to readable format"""
    lines = []
    prefix = "  " * indent
    for field_num, wire_type, value in decoded:
        if wire_type == "nested" and isinstance(value, list):
            lines.append(f"{prefix}Field {field_num} (message):")
            lines.append(protobuf_to_readable(value, indent + 1))
        else:
            lines.append(f"{prefix}Field {field_num}: {value}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("Election Data Downloader & Decoder")
    print("=" * 60)
    
    # Step 1: Download all files
    files = {}
    for name, url in URLS.items():
        try:
            files[name] = download_file(name, url)
        except Exception as e:
            print(f"[ERROR] Failed to download {name}: {e}")
    
    # Step 2: Analyze config (JSON)
    if "config" in files:
        print("\n" + "=" * 60)
        print("Config file:")
        with open(files["config"], "r", encoding="utf-8") as f:
            config = json.load(f)
        print(json.dumps(config, ensure_ascii=False, indent=2)[:2000])
    
    # Step 3: Analyze binary files
    for name in ["master", "score_69", "score_66"]:
        if name in files:
            result = analyze_binary(files[name], name)
    
    print("\n" + "=" * 60)
    print("Done! Check the data_raw/ directory for downloaded files.")
