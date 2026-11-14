"""Tests of COCO-to-YOLO conversion (synthetic fixtures, not model training)."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image

module_path=Path(__file__).resolve().parents[1]/"scripts"/"train_public_paint.py"
spec=importlib.util.spec_from_file_location("train_public_paint",module_path)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class PaintDataTests(unittest.TestCase):
    def create(self,root,categories=None):
        cats=categories or [{"id":1,"name":"paint_defect"}]
        for split,color in [("train",21),("valid",93),("test",182)]:
            folder=root/split
            folder.mkdir(parents=True)
            images=[]
            annotations=[]
            for i in range(5):
                name=f"{split}_{i}.jpg"
                Image.new("RGB",(64,64),(color,i*27,35)).save(folder/name)
                images.append({"id":i,"file_name":name,"width":64,"height":64})
                annotations.append({"id":i,"image_id":i,"category_id":1,
                                    "bbox":[3,7,25,30],"iscrowd":0})
            (folder/"_annotations.coco.json").write_text(json.dumps({
                "images":images,"annotations":annotations,"categories":cats,
            }))
    def test_conversion_preserves_independent_holdouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.create(root)
            report=mod.convert_coco(root,root/"out",min_per_split=5)
            self.assertEqual(report["splits"]["train"]["boxes"],5)
            self.assertEqual(report["splits"]["test"]["selected"],5)
            labels=list((root/"out"/"train"/"labels").glob("*.txt"))
            self.assertEqual(len(labels),5)
            self.assertTrue(all(x.read_text().startswith("0 ") for x in labels))
    def test_wrong_class_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.create(root,[{"id":1,"name":"some_unknown_feature"}])
            with self.assertRaisesRegex(ValueError,"paint_defect"):
                mod.convert_coco(root,root/"out",min_per_split=5)
    def test_cross_split_duplicates_are_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.create(root)
            source=root/"train"/"train_0.jpg"
            (root/"test"/"test_0.jpg").write_bytes(source.read_bytes())
            with self.assertRaisesRegex(ValueError,"duplicada"):
                mod.convert_coco(root,root/"out",min_per_split=5)

if __name__=="__main__":unittest.main()
