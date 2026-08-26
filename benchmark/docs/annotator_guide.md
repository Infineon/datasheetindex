# Annotator guide: blind re-derivation of the grading surface

You've been asked to independently re-derive two fields for 25 claims.
This doc tells you exactly what to do. **You don't need to read any other
project docs.** Expected time: about an hour.

---

## 1. Why this exists

The project grades an LLM agent on whether it correctly extracts spec
values from datasheets. Whether an extraction counts as *correct* is
decided by two hand-written fields on each claim:

| Field | What it does |
|---|---|
| `value_contains` | A list of substrings. The agent's answer must contain **all** of them, or the claim fails. |
| `confidence_min` | A floor. The agent's self-reported confidence must be **at least** this, or the claim fails. |

Every accuracy number in the paper depends on those two fields. They were
written by one person, early, during a period when a bug was leaking
answers into the agent's prompt. Nobody revisited them afterwards.

So the worry is **fitting**: if a substring list or a floor was chosen by
looking at what the agent happened to produce, then the test was shaped
around the answer, and re-running the agent cannot detect that.

Your job is to write down what those fields *should* be, without seeing
what they currently are. Where you and the original agree, that is
evidence the surface is sound. Where you differ, we report it — and we
check mechanically whether your version would have changed any published
result.

**There is no right answer we are hoping for.** A disagreement is a
finding, not a mistake. Please do not try to guess what we wrote.

---

## 2. Setup

The folder you were sent contains everything:

```
README.md            this guide
rederivation.yaml    the 25 claims, for you to fill in
datasheets/          the three PDFs
```

Nothing else is needed — no repository, no Python, no install. Open
`rederivation.yaml` in any text editor, find `annotator: ''` in the
`metadata:` block near the top, and put the label we gave you between
the quotes — a non-identifying one such as `annotator-1`. Please do not
put your name there: these files are published with the benchmark, and
the label is only there to tell two annotators' files apart.

---

## 3. Keep it blind

There is nothing here you need to avoid: the folder was built to contain
no answers, and that is checked automatically when it is generated. Just
work from what you were sent, and please don't ask us what we wrote until
you have finished.

Your file deliberately withholds six things: the current
`value_contains` and `confidence_min`, the page and sentence the value
came from, and the claimed numeric range. Everything you need is the
claim block plus the datasheet.

One thing you *are* given, so you know it is not an oversight: the
expected **unit** and the parameter name. The datasheet states the unit
anyway, so withholding it would not create independence. Agreement on
units is therefore reported separately from agreement on numbers, and
the numbers are what the exercise is really about.

If you look at something you shouldn't, by accident, just say so. It is
recoverable; an unreported peek is not.

---

## 4. What a claim block looks like

```yaml
  # ======================================================================
  # CLAIM: example-quiescent-current          <- illustrative, not a real claim
  # ======================================================================
  # PARAMETER:  Quiescent supply current IQ
  # DATASHEET:  datasheets/<the file named for that claim>.pdf
  # UNIT:       uA
  # CONDITIONS: VCC_V=5.0V*, temperature_C=25.0C   (* = load-bearing)
  # Stated quiescent current with the output unloaded.
  - id: 'example-quiescent-current'
    value_contains: []      # FILL IN
    confidence_min:         # FILL IN
    notes: ''
```

`* = load-bearing` marks a condition that must hold for the claim to mean
anything. It tells you which table row to read when several rows give the
same parameter under different conditions.

---

## 5. How to fill in `value_contains`

1. Open the named datasheet and find the parameter.
2. Read the value **under the stated conditions**. If several rows match
   the parameter name, the conditions decide which one.
3. Write the substrings that an answer must contain to be that value.

**What your needles are matched against.** Not the agent's prose. Each
needle is searched in a string built only from the numbers the agent
extracted plus the unit — roughly `"20.0 200.0 mV"`. Nothing else is in
there: not the sentence the agent quoted from the datasheet, not its
reasoning, not the parameter name. That rules out a whole class of
needle that looks reasonable and can never match.

Rules of thumb, from how the field is actually used:

- **Substring match, not equality.** `'2.5'` matches inside `±2.5 kOhm`
  and also inside `12.5`. Short substrings are weak; prefer the longest
  form that is still certain to appear.
- **Usually the number and the unit.** Two entries is the common case.
- **Ranges take both ends.** A 3-48 kHz range wants both numbers.
- **No prose, no labels, no padding.** Write `'12.3'`, not `'min 12.3'`;
  the word "min" is yours, not the agent's, and will never be in the
  string. For the same reason, no leading or trailing spaces — `' mVpp '`
  fails where `'mVpp'` passes.
- **A symbol only distinguishes rows if it is the unit.** Naming the row
  (`VSAT_LOW`, `VOH_MAX`) cannot work, because symbols are not in the
  matched string. Where two adjacent rows give the same parameter, look
  for a unit that separates them — `mV` excludes a row denominated in
  `V` — and otherwise rely on the numbers.
- **Do not include the `±`, `<`, or `typ`/`max` words** unless you think
  an answer omitting them should genuinely fail. That is a real judgement
  call and we want yours.

If you cannot find the parameter, leave `value_contains: []` and explain
in `notes`. **An honest abstention is a result.** Abstentions are counted
and reported separately; a guess would silently corrupt the measurement.

---

## 6. How to fill in `confidence_min`

A number between 0 and 1. **It is a threshold you set for the agent, not
a statement about you.** Every answer the agent submits carries a number
it assigns to itself, saying how sure *it* is. `confidence_min` is the
level below which you would throw that answer out *even though the
substrings matched*.

So this field is not "how confident am I that I read the datasheet
right". If you want to record that — and please do, it is useful — put
it in `notes` in words.

Two practical points, because the range matters:

- **In practice the agent reports between 0.90 and 0.99.** A floor of
  `1.0` therefore rejects every answer it has ever given, and a floor
  anywhere below 0.90 accepts every one. The interesting settings live
  in between.
- **Leaving it blank is not neutral** — it silently falls back to a
  default rather than recording your judgement. Put a number.

Most claims will not need much thought. Pick the value you would apply
by default, write it everywhere, and depart from it only when a specific
claim argues for something else — saying why in `notes` when it does.

Whether you set it high or low matters less than being consistent: what
we compare is where you departed from your own default, not your default
against ours.

---

## 7. When you're done

Save `rederivation.yaml` and send it back. That is all.

We then score it: agreement per claim and in aggregate, every
disagreement listed, and a re-run of the archived model outputs under
*your* version to see whether any published result would change. We will
tell you what it showed.

---

## Questions?

Ask the person who sent you this. Please don't resolve an ambiguity by
asking us what we wrote — if a claim is ambiguous, that is itself
something we want recorded in `notes`.
