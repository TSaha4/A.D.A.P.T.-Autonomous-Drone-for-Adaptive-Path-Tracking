"""Normalise references_doi.bib (raw DOI-registry BibTeX) into paper/references.bib for IEEEtran.bst.

Only formatting changes are made: brace-aware parsing, IEEE-relevant fields kept, month -> BibTeX macro,
DOI printed as a note (IEEEtran.bst ignores the doi field), plus a small set of non-DOI sources added by hand
(software documentation and the AAAI D* Lite paper), each flagged in a comment. Bibliographic content
(authors, titles, venues, volumes, pages, years) is taken verbatim from the registry records, except where a
correction is listed in FIXES with its reason.
"""
import html
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
MONTHS = {m.lower(): m[:3].lower() for m in ["January", "February", "March", "April", "May", "June", "July",
                                              "August", "September", "October", "November", "December"]}
KEEP = ["author", "title", "journal", "booktitle", "volume", "number", "pages", "year", "month", "publisher",
        "institution", "type", "note"]
# key -> {field: value}; every correction states why
FIXES = {
    "stentz1994optimal": {"year": "1994"},  # registry record has no year; ICRA 1994 per the record's own booktitle
    "li2025urban": {"author": "Li, Shuiquan and Zhai, Binqing and Dang, Tianjiao and Gao, Mingyue and Pan, Yashi "
                              "and Wang, Yuanshun"},  # registry swaps given/family names (OpenAlex: "Li Shuiquan")
    "snyder1987map": {"type": "Professional Paper", "number": "1395", "institution": "U.S. Geological Survey",
                      "year": "1987"},
}
MANUAL = r"""
% ---- Non-DOI sources (verified on the publisher / project pages listed in url; accessed 2026-10-06) ----
@inproceedings{koenig2002dstar,
  author = {Koenig, Sven and Likhachev, Maxim},
  title = {{D*} {Lite}},
  booktitle = {Proc. 18th National Conference on Artificial Intelligence (AAAI-02)},
  pages = {476--483},
  year = {2002},
  note = {Available: \url{https://www.ri.cmu.edu/publications/d-lite}}
}
@misc{mavlink_fileformat,
  author = {{MAVLink Development Team}},
  title = {File Formats: Mission Plain-Text File Format ({QGC WPL})},
  howpublished = {MAVLink Developer Guide. [Online]. Available: \url{https://mavlink.io/en/file_formats/}},
  note = {Accessed: Oct. 6, 2026},
  year = {2026}
}
@misc{mavlink_common,
  author = {{MAVLink Development Team}},
  title = {{MAVLink} Common Message Set (common.xml)},
  howpublished = {[Online]. Available: \url{https://github.com/mavlink/mavlink/blob/master/message_definitions/v1.0/common.xml}},
  note = {Accessed: Oct. 6, 2026},
  year = {2026}
}
@misc{mavlink_mission,
  author = {{MAVLink Development Team}},
  title = {Mission Protocol},
  howpublished = {MAVLink Developer Guide. [Online]. Available: \url{https://mavlink.io/en/services/mission.html}},
  note = {Accessed: Oct. 6, 2026},
  year = {2026}
}
@misc{ardupilot_missioncmds,
  author = {{ArduPilot Dev Team}},
  title = {Copter Mission Command List},
  howpublished = {ArduPilot Documentation. [Online]. Available: \url{https://ardupilot.org/copter/docs/mission-command-list.html}},
  note = {Accessed: Oct. 6, 2026},
  year = {2026}
}
@misc{px4_missions,
  author = {{PX4 Development Team}},
  title = {Missions},
  howpublished = {PX4 User Guide. [Online]. Available: \url{https://docs.px4.io/main/en/flying/missions}},
  note = {Accessed: Oct. 6, 2026},
  year = {2026}
}
@misc{openmeteo_docs,
  author = {{Open-Meteo}},
  title = {Weather Forecast {API} Documentation},
  howpublished = {[Online]. Available: \url{https://open-meteo.com/en/docs}},
  note = {Accessed: Oct. 6, 2026},
  year = {2026}
}
@article{bradski2000opencv,
  author = {Bradski, G.},
  title = {The {OpenCV} Library},
  journal = {Dr. Dobb's Journal of Software Tools},
  year = {2000},
  note = {No DOI; standard citation requested by the OpenCV project}
}
"""


def parse_entries(text):
    entries, i = [], 0
    while True:
        i = text.find("@", i)
        if i < 0:
            return entries
        j = text.index("{", i)
        etype = text[i + 1:j].strip().lower()
        depth, k = 0, j
        while True:
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        body = text[j + 1:k]
        key, rest = body.split(",", 1)
        entries.append((etype, key.strip(), parse_fields(rest)))
        i = k + 1


def parse_fields(s):
    fields, i, n = {}, 0, len(s)
    while i < n:
        m = re.compile(r"\s*,?\s*([A-Za-z_]+)\s*=\s*").match(s, i)
        if not m:
            break
        name, i = m.group(1).lower(), m.end()
        if i < n and s[i] == "{":
            depth, k = 0, i
            while True:
                if s[k] == "{":
                    depth += 1
                elif s[k] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            value, i = s[i + 1:k], k + 1
        elif i < n and s[i] == '"':
            k = s.index('"', i + 1)
            value, i = s[i + 1:k], k + 1
        else:
            m2 = re.compile(r"[^,}]+").match(s, i)
            value, i = m2.group(0).strip(), m2.end()
        fields[name] = value.strip()
    return fields


def main():
    raw = open(os.path.join(PAPER, "references_doi.bib"), encoding="utf-8").read()
    # The registry key for Rosenkrantz et al. contains a comma ("Lewis, II"); drop the stray token after our key
    raw = re.sub(r"(@article\{rosenkrantz1977analysis),\s*ii_1977,", r"\1,", raw)
    out = ["% A.D.A.P.T. paper bibliography. DOI entries generated from doi.org registry metadata by",
           "% experiments/fetch_bibtex.py and normalised by experiments/normalize_bib.py (formatting only).", ""]
    for etype, key, f in parse_entries(raw):
        f.update(FIXES.get(key, {}))
        if "doi" in f:
            f["note"] = "doi: " + f["doi"].replace("_", r"\_")
        if etype == "misc" and "publisher" in f and "journal" not in f:
            f["howpublished"] = f.pop("publisher")
        if "month" in f:
            abbrev = f["month"].strip("{}").lower()[:3]
            f["month"] = abbrev if abbrev in MONTHS.values() else f["month"]
        if etype == "article":
            f.pop("publisher", None)
        if key == "snyder1987map":  # USGS Professional Paper: registry types it as a book with a journal field
            etype = "techreport"
            f.pop("journal", None)
        for name in list(f):  # LaTeX-safe text: HTML entities/tags from registry, dashes, ampersands
            v = re.sub(r"</?(i|b|sub|sup|em|scp)>", "", html.unescape(f[name]))
            v = v.replace("–", "--").replace("‐", "-").replace("‑", "-")
            f[name] = re.sub(r"(?<!\\)&", r"\\&", v)
        lines = [f"@{etype}{{{key},"]
        for name in KEEP + ["howpublished"]:
            if name in f and f[name]:
                v = f[name]
                lines.append(f"  {name} = {v}," if name == "month" and v in MONTHS.values() else f"  {name} = {{{v}}},")
        lines[-1] = lines[-1].rstrip(",")
        lines.append("}")
        out.append("\n".join(lines))
    out.append(MANUAL)
    with open(os.path.join(PAPER, "references.bib"), "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(out))
    print("entries:", len(parse_entries("\n".join(out))))


if __name__ == "__main__":
    main()
