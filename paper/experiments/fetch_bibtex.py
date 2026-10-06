"""Fetch authoritative BibTeX for every DOI-cited reference via doi.org content negotiation
(Crossref/DataCite metadata), and re-key the entries. Output: paper/references_doi.bib (+ fetch log)."""
import re, sys, time, requests
KEYS = {
 "koenig2005fast": "10.1109/TRO.2004.838026", "stentz1994optimal": "10.1109/ROBOT.1994.351061",
 "hart1968formal": "10.1109/TSSC.1968.300136", "rosenkrantz1977analysis": "10.1137/0206041",
 "held1962dynamic": "10.1137/0110015", "erdelj2017help": "10.1109/MPRV.2017.11",
 "mohddaud2022applications": "10.1016/j.scijus.2021.11.002", "rahnemoonfar2021floodnet": "10.1109/ACCESS.2021.3090981",
 "bonafilia2020sen1floods11": "10.1109/CVPRW50498.2020.00113", "twele2016sentinel": "10.1080/01431161.2016.1192304",
 "mcfeeters1996ndwi": "10.1080/01431169608948714", "guidolin2016weighted": "10.1016/j.envsoft.2016.07.008",
 "ghimire2013formulation": "10.2166/hydro.2012.245", "bates2000simple": "10.1016/S0022-1694(00)00278-X",
 "jamali2019cellular": "10.1029/2018WR023679", "teng2017flood": "10.1016/j.envsoft.2017.01.006",
 "aggarwal2020path": "10.1016/j.comcom.2019.10.014", "karur2021survey": "10.3390/vehicles3030027",
 "cabreira2019survey": "10.3390/drones3010004", "otto2018optimization": "10.1002/net.21818",
 "murray2015flying": "10.1016/j.trc.2015.03.005", "dong2026path": "10.3390/drones10050322",
 "rabta2018drone": "10.1016/j.ijdrr.2018.02.020", "chowdhury2017drones": "10.1016/j.ijpe.2017.03.024",
 "rejeb2021humanitarian": "10.1016/j.iot.2021.100434", "koubaa2019micro": "10.1109/ACCESS.2019.2924410",
 "meier2012pixhawk": "10.1007/s10514-012-9281-4", "shakhatreh2019unmanned": "10.1109/ACCESS.2019.2909530",
 "khan2022emerging": "10.1002/rob.22075", "gioia2026perspective": "10.1016/j.arcontrol.2026.101054",
 "li2025urban": "10.1038/s41598-025-31599-6", "alotaibi2019lsar": "10.1109/ACCESS.2019.2912306",
 "gao2021weather": "10.1038/s41598-021-91325-w", "gebrehiwot2019deep": "10.3390/s19071486",
 "munawar2021uavs": "10.3390/su13147547", "simantiris2024unsupervised": "10.3390/rs16122126",
 "ibrahim2021application": "10.11591/ijeecs.v23.i2.pp1219-1226", "perks2016advances": "10.5194/hess-20-4005-2016",
 "patterson2014timely": "10.1016/j.imavis.2014.06.006", "kaljahi2019automatic": "10.1016/j.eswa.2019.01.024",
 "snyder1987map": "10.3133/pp1395", "battersby2014implications": "10.3138/carto.49.2.2313",
 "vincenty1975direct": "10.1179/sre.1975.23.176.88", "karney2013algorithms": "10.1007/s00190-012-0578-z",
 "suzuki1985topological": "10.1016/0734-189X(85)90016-7", "haralick1987image": "10.1109/TPAMI.1987.4767941",
 "smith1978color": "10.1145/965139.807361", "zippenfenig2024openmeteo": "10.5281/zenodo.7970649",
}
out, log = [], []
for key, doi in KEYS.items():
    for attempt in range(3):
        r = requests.get(f"https://doi.org/{doi}", headers={"Accept": "application/x-bibtex; charset=utf-8"}, timeout=60)
        if r.status_code == 200 and r.text.strip().startswith("@"):
            break
        time.sleep(1)
    if r.status_code != 200 or not r.text.strip().startswith("@"):
        log.append(f"FAIL {key} {doi} {r.status_code}"); continue
    bib = r.content.decode("utf-8").strip()
    bib = re.sub(r"^@(\w+)\{[^,]+,", lambda m: "@" + m.group(1).lower() + "{" + key + ",", bib, count=1)
    out.append(bib); log.append(f"OK   {key} {doi}")
    time.sleep(0.2)
open("references_doi.bib", "w", encoding="utf-8").write("\n\n".join(out) + "\n")
open("experiments/results/bibtex_fetch.log", "w").write("\n".join(log) + "\n")
print("\n".join(l for l in log if l.startswith("FAIL")) or "all fetched", len(out))
