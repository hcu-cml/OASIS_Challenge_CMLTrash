# Roadside trash survey: run brief

**Area** Hazaribagh, Dhaka  
**Run** `94c7703b81`

## Imagery availability

16 street-level images are available. 1 of 16 grid cells have imagery, and where imagery exists the typical density is 14 images per km².

- Source images available: 16
- Audited coverage cells: 16
- Cells containing imagery: 1
- Typical density where imagery exists: 14.16 images/km²

## Trash detections

- Original images screened for trash: 16
- Images the vision model called excessive roadside trash: 1
- Images with SAM3 trash masks: 1
- Candidate images: 1
- SAM3 trash masks: 4
- Disagreements: 1

## Privacy (SAM3-only)

- Candidate images sanitized: 1
- Images with detected humans (blurred): 1
- Images with no humans detected: 0

## Human-in-the-loop review

- Accepted observations after review: 4

## CityGML export

- Approved detections exported: 4
- File: `trash_citygml.gml` (CityGML 2.0, gen:GenericCityObject)

## Coordinates

Each exported detection is placed at the GPS position the camera recorded, which is the point the waste was observed from. This agent does not infer a position for the waste itself: doing so from a single uncalibrated street-level photograph required a depth estimate, a compass reading and an assumed field of view, and the result could not be validated. That inference was removed rather than reported with false precision.
