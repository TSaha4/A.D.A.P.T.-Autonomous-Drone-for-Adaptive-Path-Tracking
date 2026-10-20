"""Manuscript audit: citations vs bibliography, labels vs references, abstract length, element counts,
and an inventory of every number appearing in the LaTeX sources (for the claim-evidence audit).
Output: printed report + results/manuscript_numbers.txt
"""
import glob
import io
import os
import re

PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PAPER)
files = ["main.tex"] + sorted(p.replace("\\", "/") for p in glob.glob("sections/*.tex") + glob.glob("figures/*.tex"))
tex = {f: io.open(f, encoding="utf-8").read() for f in files}
body = "\n".join(tex.values())

cited = set()
for m in re.finditer(r"\\cite\{([^}]*)\}", body):
    cited.update(k.strip() for k in m.group(1).split(","))
bib = set(re.findall(r"^@\w+\{([^,]+),", io.open("references.bib", encoding="utf-8").read(), re.M))
print("cited keys:", len(cited), "| bibliography entries:", len(bib))
print("cited but missing from bib:", sorted(cited - bib))
print("in bib but not cited (not printed by BibTeX):", sorted(bib - cited))

labels = set(re.findall(r"\\label\{([^}]*)\}", body))
refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]*)\}", body))
print("references without label:", sorted(refs - labels))
print("labels never referenced:", sorted(labels - refs))

ab = tex["sections/abstract.tex"].split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
ab = re.sub(r"\\adapt\{\}", "ADAPT", ab)
ab = re.sub(r"\$[^$]*\$", "X", ab)
ab = re.sub(r"\\[a-zA-Z]+|[{}~]", " ", ab)
print("abstract words:", len(ab.split()))
kw = re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", tex["sections/abstract.tex"], re.S).group(1)
print("index terms:", len([k for k in kw.split(",") if k.strip()]))

print("figures:", len(re.findall(r"\\begin\{figure\*?\}", body)), "| tables:", len(re.findall(r"\\begin\{table\*?\}", body)),
      "| algorithms:", len(re.findall(r"\\begin\{algorithm\}", body)))
eq = len(re.findall(r"\\begin\{(?:equation|multline)\}", body))
eq += sum(s.count("\\\\") + 1 for s in re.findall(r"\\begin\{align\}(.*?)\\end\{align\}", body, re.S))
print("numbered equations:", eq)

with open(os.path.join("experiments", "results", "manuscript_numbers.txt"), "w", encoding="utf-8") as out:
    for f, s in tex.items():
        for i, line in enumerate(s.splitlines(), 1):
            nums = re.findall(r"(?<![\w\\{])\d[\d.,\\]*\d|\b\d\b", line)
            nums = [n for n in nums if not re.fullmatch(r"\d", n)]
            if nums:
                out.write(f"{f}:{i}: {sorted(set(nums))}\n")
print("number inventory written to experiments/results/manuscript_numbers.txt")
