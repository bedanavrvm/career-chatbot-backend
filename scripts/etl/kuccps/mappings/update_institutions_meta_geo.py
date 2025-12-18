import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPINGS = ROOT / "mappings"
PROCESSED = ROOT / "processed"

meta_fp = MAPPINGS / "institutions_meta.csv"
inst_fp = PROCESSED / "institutions.csv"
geo_fp = MAPPINGS / "institutions_geo.csv"
camp_fp = MAPPINGS / "institutions_campuses.csv"

def read_existing_meta():
    existing_by_code = {}
    existing_by_name = {}
    if meta_fp.exists():
        with meta_fp.open(encoding='utf-8') as f:
            r = csv.DictReader(f)
            for row in r:
                code = (row.get('institution_code') or '').strip()
                name_contains = (row.get('name_contains') or row.get('name') or '').strip()
                alias = (row.get('alias') or '').strip()
                website = (row.get('website') or '').strip()
                name = (row.get('name') or '').strip()
                if code:
                    existing_by_code[code] = {'alias': alias, 'website': website, 'name': name}
                if name_contains:
                    existing_by_name[name_contains.upper()] = {'alias': alias, 'website': website}
    return existing_by_code, existing_by_name

existing_by_code, existing_by_name = read_existing_meta()

# Curated alias/website additions (official sites) for fallback when not present in processed CSV
additions = [
    ('RONGO UNIVERSITY','RU','https://www.rongovarsity.ac.ke'),
    ('KISII UNIVERSITY','KU','https://kisiiuniversity.ac.ke'),
    ('CHUKA UNIVERSITY','CU','https://www.chuka.ac.ke'),
    ('UNIVERSITY OF KABIANGA','UK','http://www.kabianga.ac.ke'),
    ('MASENO UNIVERSITY','MU','https://www.maseno.ac.ke'),
    ('UNIVERSITY OF EMBU','UE','https://embuni.ac.ke'),
    ('KCA UNIVERSITY','KCAU','https://www.kcau.ac.ke'),
    ('GRETSA UNIVERSITY','GU','https://gretsauniversity.ac.ke'),
    ('SCOTT CHRISTIAN UNIVERSITY','SCU','https://scott.ac.ke'),
    ('ST PAULS UNIVERSITY','SPU','https://www.spu.ac.ke'),
    ('KIBABII UNIVERSITY','KIBU','https://kibu.ac.ke'),
    ('UNIVERSITY OF ELDORET','UOELD','https://www.uoeld.ac.ke'),
    ('UZIMA UNIVERSITY','UU','https://uzimauniversity.ac.ke'),
    ('UNIVERSITY OF EASTERN AFRICA, BARATON','UEAB','https://ueab.ac.ke'),
    ('GREAT LAKES UNIVERSITY OF KISUMU','GLUK','https://www.gluk.ac.ke'),
    ('PRESBYTERIAN UNIVERSITY OF EAST AFRICA','PUEA','https://puea.ac.ke'),
    ('INTERNATIONAL LEADERSHIP UNIVERSITY','ILU','https://kenya.ilu.edu'),
    ('KENYA HIGHLANDS EVANGELICAL UNIVERSITY','KHU','https://khu.ac.ke'),
    ('DEDAN KIMATHI UNIVERSITY OF TECHNOLOGY','DEKUT','https://www.dkut.ac.ke'),
    ('LAIKIPIA UNIVERSITY','LU','https://laikipia.ac.ke'),
    ('KARATINA UNIVERSITY','KARU','https://karu.ac.ke'),
    ("MURANG'A UNIVERSITY OF TECHNOLOGY",'MUT','https://www.mut.ac.ke'),
    ('ZETECH UNIVERSITY','ZU','https://www.zetech.ac.ke'),
    ('KIRIRI WOMENS UNIVERSITY OF SCIENCE AND TECHNOLOGY','KWUST','https://kwust.ac.ke'),
    ('KAIMOSI FRIENDS UNIVERSITY','KAFU','https://kafu.ac.ke'),
    ('TANGAZA UNIVERSITY','TU','https://tangaza.ac.ke'),
    ('CATHOLIC UNIVERSITY OF EASTERN AFRICA','CUEA','https://www.cuea.edu'),
    ('TOM MBOYA UNIVERSITY','TMU','https://tmu.ac.ke'),
    ('AMREF INTERNATIONAL UNIVERSITY','AMIU','https://amref.ac.ke'),
    ('RIARA UNIVERSITY','RU','https://riarauniversity.ac.ke'),
    ('MANAGEMENT UNIVERSITY OF AFRICA','MUA','https://mua.ac.ke'),
    ('PAN AFRICA CHRISTIAN UNIVERSITY','PACU','https://www.pacuniversity.ac.ke'),
    ('KENYA METHODIST UNIVERSITY','KEMU','https://kemu.ac.ke'),
    ('AFRICA NAZARENE UNIVERSITY','ANU','https://www.anu.ac.ke'),
    ('JARAMOGI OGINGA ODINGA UNIVERSITY OF SCIENCE AND TECHNOLOGY','JOOUST','https://www.jooust.ac.ke'),
    ('TAITA TAVETA UNIVERSITY','TTU','https://www.ttu.ac.ke'),
    ('KABARAK UNIVERSITY','KABU','https://kabarak.ac.ke'),
    ('KAG EAST UNIVERSITY','KAGEU','https://kag.east.ac.ke'),
    ('MAMA NGINA UNIVERSITY COLLEGE','MNUC','https://mnu.ac.ke'),
    ('KOITALEEL SAMOEI UNIVERSITY COLLEGE','KSUC','https://ksu.ac.ke'),
    ('ALUPE UNIVERSITY','AU','https://au.ac.ke'),
    ('TURKANA UNIVERSITY COLLEGE','TUC','https://tuc.ac.ke'),
    ('ISLAMIC UNIVERSITY OF KENYA','IUK','https://iuk.ac.ke'),
    ('OPEN UNIVERSITY OF KENYA','OUK','https://ouk.ac.ke'),
    ('THE EAST AFRICAN UNIVERSITY','TEAU','https://teau.ac.ke'),
    ('LUKENYA UNIVERSITY','LU','https://lukenyauniversity.ac.ke'),
    ('MARIST INTERNATIONAL UNIVERSITY COLLEGE','MIUC','https://www.miuc.ac.ke'),
    ('AFRICA INTERNATIONAL UNIVERSITY','AIU','https://www.aiu.ac.ke'),
]

additions_map = {name.upper(): {'alias': alias, 'website': site} for name, alias, site in additions}

# Load processed institutions for authoritative code/name/alias/website
inst_rows = []
name_by_code = {}
alias_by_code = {}
website_by_code = {}
if inst_fp.exists():
    with inst_fp.open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            code = (r.get('code') or '').strip()
            name = (r.get('name') or '').strip()
            alias = (r.get('alias') or '').strip()
            website = (r.get('website') or '').strip()
            if code and name:
                inst_rows.append(r)
                name_by_code[code] = name
                alias_by_code[code] = alias
                website_by_code[code] = website

# Rewrite institutions_meta.csv with extended header and one row per institution code
with meta_fp.open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['institution_code','name','name_contains','alias','website'])
    for code in sorted(name_by_code.keys()):
        name = name_by_code[code]
        # Merge precedence: existing_by_code > additions_map (by name) > processed alias/website > blanks
        ex = existing_by_code.get(code, {})
        add = additions_map.get(name.upper(), {})
        alias = ex.get('alias') or alias_by_code.get(code) or add.get('alias') or ''
        website = ex.get('website') or website_by_code.get(code) or add.get('website') or ''
        # name_contains left blank (code matching will be used primarily); keep for manual substring overrides if needed
        w.writerow([code, name, '', alias, website])

# Counties per institution code inferred from public sources
county_by_code = {
    '1053':'Siaya',
    '1057':'Nakuru',
    '1060':'Nairobi',
    '1061':'Nakuru',
    '1063':'Mombasa',
    '1066':'Nairobi',
    '1068':'Nairobi',
    '1073':'Migori',
    '1077':'Meru',
    '1078':'Kajiado',
    '1079':'Kirinyaga',
    '1080':'Nairobi',
    '1082':'Kakamega',
    '1087':'Kisii',
    '1088':'Kiambu',
    '1090':'Machakos',
    '1091':'Taita Taveta',
    '1093':'Embu',
    '1096':'Garissa',
    '1103':'Nairobi',
    '1105':'Tharaka-Nithi',
    '1107':'Kiambu',
    '1108':'Bungoma',
    '1111':'Nairobi',
    '1112':'Nairobi',
    '1114':'Uasin Gishu',
    '1116':'Kisumu',
    '1117':'Kilifi',
    '1118':'Kericho',
    '1119':'Nairobi',
    '1162':'Machakos',
    '1164':'Nairobi',
    '1165':'Narok',
    '1166':'Kitui',
    '1169':'Kericho',
    '1170':'Machakos',
    '1173':'Nyeri',
    '1176':'Laikipia',
    '1181':'Nandi',
    '1192':'Kisumu',
    '1196':'Kiambu',
    '1225':'Nairobi',
    '1229':'Kisumu',
    '1240':'Meru',
    '1244':'Nyeri',
    '1246':"Murang'a",
    '1249':'Kiambu',
    '1253':'Uasin Gishu',
    '1263':'Nairobi',
    '1279':'Kiambu',
    '1425':'Kiambu',
    '1460':'Nairobi',
    '1470':'Vihiga',
    '1475':'Nairobi',
    '1480':'Nairobi',
    '1495':'Machakos',
    '1500':'Kajiado',
    '1515':'Homa Bay',
    '1530':'Nairobi',
    '1555':'Nairobi',
    '1570':'Turkana',
    '1580':'Kiambu',
    '1600':'Busia',
    '1685':'Tharaka-Nithi',
    '1700':'Bomet',
    '3890':'Nandi',
    '3895':'Kajiado',
    '4275':'Kiambu',
    '5145':'Machakos',
}

region_by_county = {
    'Nairobi':'Nairobi',
    'Kiambu':'Central',
    "Murang'a":'Central',
    'Nyeri':'Central',
    'Kirinyaga':'Central',
    'Laikipia':'Rift Valley',
    'Nakuru':'Rift Valley',
    'Uasin Gishu':'Rift Valley',
    'Nandi':'Rift Valley',
    'Kericho':'Rift Valley',
    'Bomet':'Rift Valley',
    'Baringo':'Rift Valley',
    'Narok':'Rift Valley',
    'Turkana':'Rift Valley',
    'Embu':'Eastern',
    'Meru':'Eastern',
    'Tharaka-Nithi':'Eastern',
    'Machakos':'Eastern',
    'Makueni':'Eastern',
    'Kitui':'Eastern',
    'Kajiado':'Rift Valley',
    'Mombasa':'Coast',
    'Kilifi':'Coast',
    'Kwale':'Coast',
    'Taita Taveta':'Coast',
    'Tana River':'Coast',
    'Lamu':'Coast',
    'Kisumu':'Nyanza',
    'Siaya':'Nyanza',
    'Homa Bay':'Nyanza',
    'Kisii':'Nyanza',
    'Migori':'Nyanza',
    'Nyamira':'Nyanza',
    'Kakamega':'Western',
    'Bungoma':'Western',
    'Busia':'Western',
    'Vihiga':'Western',
    'Garissa':'North Eastern',
    'Wajir':'North Eastern',
    'Mandera':'North Eastern',
}

# Write geo rows for all processed institutions (with name for readability)
codes = sorted(name_by_code.keys())
with geo_fp.open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['institution_code','name','region','county'])
    for c in codes:
        county = county_by_code.get(c, '')
        region = region_by_county.get(county, '') if county else ''
        w.writerow([c, name_by_code.get(c, ''), region, county])

prog_fp = PROCESSED / 'programs.csv'
campus_map = {}
if prog_fp.exists():
    with prog_fp.open(encoding='utf-8') as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            code = (row.get('institution_code') or '').strip()
            campus = (row.get('campus') or '').strip()
            if not code:
                continue
            if campus:
                campus_map.setdefault(code, set()).add(campus)

# Seed known non-main campuses with geo for select institutions
seed_campuses = {
    '1263': ['PARKLANDS','CHIROMO','LOWER KABETE','KIKUYU','KISUMU','MOMBASA'],  # UON
    '1249': ['NAIROBI CBD','KAREN','NAKURU','KISII','MOMBASA','KISUMU','ELDORET'],  # JKUAT
    '1279': ['NAIROBI CBD','NAKURU','ELDORET','MOMBASA','KISII'],  # MKU
}
seed_geo = {
    ('1263','PARKLANDS'): ('Nairobi','Nairobi','Nairobi'),
    ('1263','CHIROMO'): ('Nairobi','Nairobi','Nairobi'),
    ('1263','LOWER KABETE'): ('Nairobi','Nairobi','Nairobi'),
    ('1263','KIKUYU'): ('Kikuyu','Kiambu','Central'),
    ('1263','KISUMU'): ('Kisumu','Kisumu','Nyanza'),
    ('1263','MOMBASA'): ('Mombasa','Mombasa','Coast'),
    ('1249','NAIROBI CBD'): ('Nairobi','Nairobi','Nairobi'),
    ('1249','KAREN'): ('Karen','Nairobi','Nairobi'),
    ('1249','NAKURU'): ('Nakuru','Nakuru','Rift Valley'),
    ('1249','KISII'): ('Kisii','Kisii','Nyanza'),
    ('1249','MOMBASA'): ('Mombasa','Mombasa','Coast'),
    ('1249','KISUMU'): ('Kisumu','Kisumu','Nyanza'),
    ('1249','ELDORET'): ('Eldoret','Uasin Gishu','Rift Valley'),
    ('1279','NAIROBI CBD'): ('Nairobi','Nairobi','Nairobi'),
    ('1279','NAKURU'): ('Nakuru','Nakuru','Rift Valley'),
    ('1279','ELDORET'): ('Eldoret','Uasin Gishu','Rift Valley'),
    ('1279','MOMBASA'): ('Mombasa','Mombasa','Coast'),
    ('1279','KISII'): ('Kisii','Kisii','Nyanza'),
}

with camp_fp.open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    # Non-main branches only; main is defined in institutions_geo.csv
    w.writerow(['institution_code','campus','town','county','region'])
    total_rows = 0
    for code in sorted(name_by_code.keys()):
        campuses_src = set(campus_map.get(code, set()))
        if code in seed_campuses:
            campuses_src.update(seed_campuses[code])
        campuses = sorted(campuses_src)
        # Filter out MAIN markers
        non_main = [c for c in campuses if c.strip().upper() not in {'MAIN', 'MAIN CAMPUS'}]
        for cName in non_main:
            key = (code, cName.strip().upper())
            town, county, region = seed_geo.get(key, ('', '', ''))
            w.writerow([code, cName, town, county, region])
            total_rows += 1

print(f"Wrote {len(codes)} institutions: meta+geo updated; campuses (non-main) template generated with {total_rows} rows.")
