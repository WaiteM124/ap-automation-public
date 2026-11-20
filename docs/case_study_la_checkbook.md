# Case study: coding 23,000 real invoice lines with a self-learning AP engine

**A replay of the City of Los Angeles' public payments data, in shadow mode, on months the system had never seen.**

## At a glance

| | |
|---|---|
| **Data** | 22,990 real payment lines (30,000-line sample, 120 vendors, 243 account codes, 48 departments) from Checkbook L.A., fiscal year 2026 |
| **Mode** | Shadow: the system suggests a code for each line and is scored against the account code the City actually used. Nothing is posted |
| **Result on unseen months** | **71.5% of lines coded automatically at 99.5% accuracy** (42 wrong out of 7,949) |
| **Starting point** | 60.2% automated at 92.3% accuracy before tuning |
| **Cost** | Embedding cost for the first full replay: about $0.003 |

## The question

Non-PO invoices are coded by hand: someone reads the description, recognises the vendor, and picks a GL account and cost centre. It is slow, inconsistent between people, and hard to audit. We built an engine that proposes the code from the company's own history, and decides when it is safe to act and when a human must look.

The question we wanted answered with real data, not a demo: **how often can it be trusted, and does it know when it can't be?**

## How the engine works

- **Vendor-scoped retrieval.** For each invoice it finds that vendor's past, human-confirmed lines that look most similar, using text embeddings.
- **A rule gate.** It only proposes an automatic code when there is enough history and the history agrees. Everything else goes to a person.
- **Learning from people only.** The system learns from human confirmations and corrections. It never learns from its own automatic postings.
- **Full audit.** Every decision is recorded with the evidence behind it.

## The test

We used the City of Los Angeles Controller's open payments data (Checkbook L.A.), because it is real, large, and public, so anyone can check our work.

- 515,904 FY2026 lines matched our filter. After removing duplicates, privacy-masked payees and empty descriptions, 452,651 remained.
- We sampled the busiest 120 vendors (250 lines each) and replayed them in date order: 6,363 lines from the first three months as history, then 22,990 lines replayed one at a time.
- 10 vendors were held out of the history entirely, to test how the system behaves with vendors it has never seen.
- The answer key is the account code the City recorded. Fields that give the answer away (account name, expenditure type, item code) were excluded, and a guard in the code checks that they are never used.
- The replay ran on a single workstation using an in-process vector store in place of our hosted Qdrant database. The retrieval, scoring and gating code is the production code. On a 1,200-line cross-check, the stand-in matched Qdrant's local mode on 95% of lines and on aggregate metrics to within about 2 points.
- Tuning used October to February. **The final numbers below are from March to June, which nothing was tuned on** (11,118 lines).

## Results

On the held-out months, in shadow mode, using OpenAI embeddings:

| Configuration | Lines coded automatically | Accuracy of those lines | Wrong |
|---|---|---|---|
| First version, default gate | 60.2% | 92.3% | 518 |
| Tuned, targeting 99% | 77.8% | 99.0% | 90 |
| **Tuned, targeting 99.5%** | **71.5%** | **99.5%** | **42** |

Accuracy here means: of the lines the system would have posted on its own, how many matched the account code the City used. The rest go to a person, and the system can be set to trade volume for accuracy, as the last two rows show.

## What we learned

- **Context was the biggest lever.** Holding accuracy at about 99%, using the department and fund as retrieval signals let the system code about 68% of lines. Making the gate stricter without adding context coded only about 35%.
- **Vendors with one code are easy; vendors with many are the work.** Vendors that always use one account auto-post at about 99.9% with no errors. The gains, and the risk, are in the 89% of vendors here that use more than one code.
- **A system that never checks its own work drifts.** In a simulation where the system acted on its own decisions and nobody corrected the mistakes, accuracy fell from about 91% in October to about 80% in June. The cause was uncorrected automatic postings, not the data changing. A sample audit of automatic postings stops this, which is why we recommend running in shadow mode first and keeping a standing audit sample afterwards.
- **Its confidence is meaningful.** On the first run its confidence scores were close to its real hit rate (calibration error 0.042). For a vendor it has never seen, a person reviews the first few lines before the system starts coding that vendor itself.

## What this does and doesn't show

This is public-sector data, and most of these payments (98%) carry a purchase order, which fixes much of the coding upstream. It shows that the engine works on real, messy, high-volume data and behaves safely. **It does not predict accuracy on your invoices.** Your vendors, your chart of accounts and your coding habits will decide that, which is why every deployment starts the same way.

## How we apply this with a client

1. **Shadow mode.** We run on your historical invoices and then on live ones, with a clerk coding as normal. You see the numbers on your own data before anything is automated.
2. **Agree the bar.** We set the accuracy target with you, and the system automates only what clears it.
3. **Go live in steps.** Automatic posting starts with the safest vendors and expands as the measured accuracy holds. An audit sample stays on permanently.

Every automatic posting and every human correction is recorded and can be reviewed.

---

*Source data: Checkbook L.A., City of Los Angeles (controllerdata.lacity.org, dataset pggv-e4fn), licensed CC BY 4.0. This case study is our own analysis; the City of Los Angeles has not reviewed or endorsed it.*
