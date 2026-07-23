from __future__ import annotations

import zipfile

from tools.data.audit_llvip_archive import audit


def _xml(record_id: str, xmax: int) -> str:
    return f"""<?xml version="1.0"?>
<annotation>
  <filename>{record_id}.jpg</filename>
  <size><width>10</width><height>8</height><depth>1</depth></size>
  <object>
    <name>person</name><difficult>0</difficult>
    <bndbox><xmin>1</xmin><ymin>1</ymin><xmax>{xmax}</xmax><ymax>7</ymax></bndbox>
  </object>
</annotation>
"""


def test_llvip_audit_checks_pairs_groups_and_boxes(tmp_path):
    archive = tmp_path / "llvip.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for split, record_id in (("train", "010001"), ("test", "020001")):
            handle.writestr(f"LLVIP/visible/{split}/{record_id}.jpg", b"visible")
            handle.writestr(f"LLVIP/infrared/{split}/{record_id}.jpg", b"infrared")
            handle.writestr(
                f"LLVIP/Annotations/{record_id}.xml",
                _xml(record_id, xmax=9 if split == "train" else 11),
            )

    result = audit(archive, class_name="person", example_limit=10)

    assert result["pairing"] == {
        "visible_infrared_train_ids_equal": True,
        "visible_infrared_test_ids_equal": True,
        "annotation_ids_equal_image_pair_ids": True,
        "cross_split_record_overlap_count": 0,
    }
    assert result["group_prefix_audit"]["total_group_count"] == 2
    assert result["group_prefix_audit"]["cross_split_group_overlap"] == []
    assert result["annotations"]["class_object_counts"] == {"person": 2}
    assert result["annotations"]["class_positive_group_counts"] == {"person": 2}
    assert result["annotations"]["invalid_bbox_count"] == 1
