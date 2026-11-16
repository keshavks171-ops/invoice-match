# Submission checklist

Deadline: **Sunday, November 15, 11:59 PM ET.** You can resubmit until then; judges see
the latest version. Submit a first version as soon as the links work.

## 1. Code link (required): viewable without signing in

The event rules say a project built before the hackathon cannot be extended, and this
repository also contains FinWise. Publish Invoice Match on its own so judges see only
the hackathon work:

```bash
# from the root of the finwise checkout, on branch claude/invoice-match-hackathon-74z2ld
git subtree split --prefix=invoice-match -b invoice-match-only
# create an empty PUBLIC repository named invoice-match on GitHub, then:
git push https://github.com/<your-account>/invoice-match.git invoice-match-only:main
```

- [ ] New public repository created and pushed (the README renders with the screenshots)
- [ ] Opened the repository link in a private/incognito window: it loads without signing in
- [ ] Last resort only: make the existing repository public and submit the folder link
      `.../tree/claude/invoice-match-hackathon-74z2ld/invoice-match`. Its history shows a
      personal email address and the earlier FinWise project, which works against blind
      review and the build-during-the-weekend rule. Prefer the split repository

## 2. Write-up (required): about one page

`submission/WRITEUP.md` follows the event template (The problem / What we built / What
we added / How it works / Results / What we'd do next / Sources) and contains the
required sentence "This project is a prototype and is not financial advice."

- [ ] Choose one: link the file on GitHub, or paste it into a Google Doc shared as
      "anyone with the link can view", or upload a PDF/Word export on the form
- [ ] Opened that link in a private window
- [ ] The **Team** line: Round 1 is reviewed blind, so the write-up points to the form.
      The event template shows `**Team:** names`; add names there only if you prefer

## 3. Submission form fields

- [ ] Track: **AI + Finance**
- [ ] Every team member's name, email and school, spelled correctly (1 to 3 high school
      students; everyone listed must have contributed)
- [ ] "What you added": our own idea; own user (volunteer treasurer), own synthetic data
      and a case-by-case challenge set, own questions (duplicates caught in 20 reviews;
      are freelancers flagged more often than companies)
- [ ] Credits, with links (copy from the write-up's Sources): scikit-learn, RapidFuzz,
      pandas, NumPy, Streamlit, LightGBM (challenger only), pytest, Playwright, Instrument
      Sans font, OWASP CSV Injection page, and **Claude (Anthropic, via Claude Code)** as
      the AI tool. No external dataset, no pretrained model, no tutorial code
- [ ] If you used any other AI tool or source yourself (for example to prepare the
      video), add it

## 4. Demo video (optional, recommended): up to 2 minutes

- [ ] Record with `submission/VIDEO_SCRIPT.md` (screen recording plus voice). Say only
      what is true for you; add your own nonprofit experience in your own words if you like
- [ ] Upload to YouTube as Unlisted, or Google Drive with "anyone with the link can view"
- [ ] Opened the link in a private window

## 5. Before you press submit

- [ ] Ran it yourself: `pip install -r requirements.txt`, `streamlit run app.py`,
      Load demo, the three demo cases, Benchmark page
- [ ] Read `matcher.py`, `evaluate.py` and the "Three technical points" in
      `submission/FINALS_OUTLINE.md`: the rules require that the team understands and
      can explain every part of the project
- [ ] The AI-assistance sentence in the write-up matches what actually happened. Edit it
      if your own contribution was larger or different
- [ ] No names, school or organization names in the README or write-up (blind review)
- [ ] Vendor names: "OC Event Rentals" was specified as fictional; businesses with
      similar names may exist. Rename it in `synth.py` (demo cases) if you want no doubt

If the form fails near the deadline, email contact@veritasai.com right away with the
code link and the write-up link.

## Dates

- Shortlist and finalists emailed: Wednesday, November 18
- Live finals: Saturday, November 21, 11:00 AM ET (see `FINALS_OUTLINE.md`)
