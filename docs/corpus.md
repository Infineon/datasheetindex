# The measurement corpus

Twenty-five publicly downloadable datasheets and technical reference manuals,
from six vendors, spanning 5 to 784 pages. Every measured claim in this
repository outside `benchmark/` is scored against this set.

The PDFs are **not redistributed here** -- they are the vendors' documents, not
ours. They are identified below by name, page count and SHA-256 so a fetched
copy can be checked against the one we measured. Put them in a directory,
named as in the first column, and point `scripts/token_economy.py --corpus` at
it.

The set is deliberately public-vendor only, which is also its main limitation:
it says nothing about internal documents, and any claim about those needs its
own measurement. It spans small parts (a 5-page diode) through large reference
manuals (a 784-page SoC TRM) because document size is the variable most of
these measurements turn on.

| Name | Pages | SHA-256 (first 16) | Source |
|---|---:|---|---|
| `vishay_1n4001` | 5 | `56a77c6615c90c11` | https://www.vishay.com/docs/88503/1n4001.pdf |
| `raspi_pico` | 31 | `757ff485227493b9` | https://datasheets.raspberrypi.com/pico/pico-datasheet.pdf |
| `ti_ina219` | 38 | `58004eda854d0747` | https://www.ti.com/lit/ds/symlink/ina219.pdf |
| `ti_ne555` | 39 | `dae2be1a919b0cfe` | https://www.ti.com/lit/ds/symlink/ne555.pdf |
| `ti_sn74hc595` | 41 | `215816945704e561` | https://www.ti.com/lit/ds/symlink/sn74hc595.pdf |
| `ti_tcan1044a` | 42 | `f2301a3f989605fe` | https://www.ti.com/lit/ds/symlink/tcan1044a-q1.pdf |
| `ti_lm317` | 44 | `323ec64a58515090` | https://www.ti.com/lit/ds/symlink/lm317.pdf |
| `ti_opa2340` | 48 | `8099f9f9e8e49c45` | https://www.ti.com/lit/ds/symlink/opa2340.pdf |
| `bosch_bmp280` | 49 | `473ff27d9df698b4` | https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmp280-ds001.pdf |
| `ti_tps54360` | 50 | `c2743e51e9f180e3` | https://www.ti.com/lit/ds/symlink/tps54360.pdf |
| `ti_ads1115` | 57 | `5fe00cf9509daba7` | https://www.ti.com/lit/ds/symlink/ads1115.pdf |
| `bosch_bme280` | 60 | `a2ccdb449fec9438` | https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bme280-ds002.pdf |
| `ti_lm358` | 68 | `ef646b009b0524a8` | https://www.ti.com/lit/ds/symlink/lm358.pdf |
| `ti_cc2640r2f` | 72 | `8403b9d742558c76` | https://www.ti.com/lit/ds/symlink/cc2640r2f.pdf |
| `esp32c3_ds` | 76 | `833fc000b4b3c3d3` | https://www.espressif.com/sites/default/files/documentation/esp32-c3_datasheet_en.pdf |
| `esp32_ds` | 78 | `a7917e6b47528c9d` | https://www.espressif.com/sites/default/files/documentation/esp32_datasheet_en.pdf |
| `infineon_psc3` | 90 | `70f1e5a495b3982d` | PSOC&trade; Control C3 (PSC3P5xD, PSC3M5xD), by part number from https://www.infineon.com/ |
| `ti_tlv9061` | 99 | `c37698e10c1c9c3f` | https://www.ti.com/lit/ds/symlink/tlv9061.pdf |
| `esp8266_trm` | 111 | `25f08aa786724a9c` | https://www.espressif.com/sites/default/files/documentation/esp8266-technical_reference_en.pdf |
| `bosch_bmi160` | 114 | `0c3dca28517322bd` | https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi160-ds000.pdf |
| `ti_msp430f5529` | 143 | `50e73041cb1f48be` | https://www.ti.com/lit/ds/symlink/msp430f5529.pdf |
| `micro_atmega328` | 294 | `fb84858d0b12c695` | https://ww1.microchip.com/downloads/en/DeviceDoc/Atmel-7810-Automotive-Microcontrollers-ATmega328P_Datasheet.pdf |
| `micro_pic16f887` | 322 | `8cca4a12d17d06af` | https://ww1.microchip.com/downloads/en/DeviceDoc/40001291H.pdf |
| `raspi_rp2040` | 642 | `be56fbb75ba0ae9e` | https://datasheets.raspberrypi.com/rp2040/rp2040-datasheet.pdf |
| `esp32_trm` | 784 | `4ba58e9fa0405ec2` | https://www.espressif.com/sites/default/files/documentation/esp32_technical_reference_manual_en.pdf |

**If a checksum does not match, the vendor has reissued the document.** That is
expected over time and is not a failure. It does mean a re-run may legitimately
differ from the published numbers, in the same way and for the same reason as
the benchmark corpus -- see
[`benchmark/docs/reproducing.md`](../benchmark/docs/reproducing.md), which
documents its own four documents this way.
