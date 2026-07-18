# Audit datasets

## 2019 Ridgecrest earthquake sequence

`ridgecrest_2019_m2_5.csv` is a frozen temporal subset of the USGS Ridgecrest
earthquake catalog. It contains 2,062 earthquakes of magnitude 2.5 or greater
from 2019-07-04 through 2019-07-12 in the bounding box 34°N–37°N,
119°W–116°W.

Source query:

<https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2019-07-04&endtime=2019-07-12&minlatitude=34&maxlatitude=37&minlongitude=-119&maxlongitude=-116&minmagnitude=2.5&orderby=time-asc>

The repository copy retains only elapsed UTC seconds, preferred magnitude, and
USGS event ID. It was retrieved on 2026-07-18. Catalog records can be revised,
so the frozen files are identified by SHA-256:

- Original USGS CSV: `6b3396b647b537074dbe7d014c3d9f1ad3e5945227495465303c64846959910b`
- Repository subset: `fe849a491b44e9b0c2fad6487c4a5fa40532ad8565b78387ebf5444e56c27d45`

Credit: U.S. Geological Survey. USGS-authored data are in the U.S. public
domain; see <https://www.usgs.gov/faqs/are-usgs-reportspublications-copyrighted>.

The magnitude threshold reduces short-term catalog incompleteness but does not
eliminate it, especially immediately after the two main shocks. The catalog is
appropriate for regression and model auditing, not a claim of a complete
seismological analysis.

Run the chronological audit with:

```sh
cargo run --release --example audit_ridgecrest
```
