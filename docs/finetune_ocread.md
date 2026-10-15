# Fine-tuning PaddleOCR's recognizer for OCRead

PaddleOCR's general recognizer misreads characters a cap rib runs through
("4113 08:26" -> "08:25", "3030" -> "2030") and mixes up lines that touch.
Fine-tuning retrains it on images of this printer's dot print with exactly
those situations. Only OCRead uses the fine-tuned model; the OCR pipeline
keeps the general one.

## 1. Make the training data (laptop)

Needs debug captures (`ocr.dynamic_roi.debug_save: true` for a session); the
cap areas around the codes are used as backgrounds.

    python -m QC_tools.make_rec_dataset

Writes `data/ocread_rec_dataset/` and `data/ocread_rec_dataset.zip`
(20 000 training + 2 000 validation line images; `--train/--val` to change).
Besides the codes it contains moulded marks (PE/PA, PP, HDPE, recycling
triangles) and rib edges right beside the code - labelled with only the code,
so the model learns to look past them - and some embossing-only lines
labelled with their letters. The output alphabet includes capital letters:
a digits-only model turned "PE/PA" into "BB/90" and a rib edge into an extra
"1" ("4113 08:261") on the line.
None of your real bottles' codes are in it: they stay a fair test.

## 2. Train (Google Colab, free GPU)

Open `QC_tools/finetune_paddleocr_rec.ipynb` in https://colab.research.google.com
(File -> Upload notebook), set Runtime -> Change runtime type -> GPU, run the
cells top to bottom, upload the dataset zip when asked. The last cell
downloads `ocread_rec_model.zip`.

### No Colab GPU available? Use Kaggle

`QC_tools/finetune_paddleocr_rec_kaggle.ipynb` does the same on
https://www.kaggle.com (free GPU, about 30 GPU hours a week):
1. Create -> New Dataset -> upload `data/ocread_rec_dataset.zip`.
2. Create -> New Notebook -> File -> Import Notebook -> the .ipynb above.
3. Session options: Accelerator *GPU T4 x2* or *GPU P100*, Internet *On*;
   Add Input -> your dataset.
4. Run All; download `ocread_rec_model.zip` from the link at the end
   (or the Output panel).

## 3. Use it in the app

Unzip it into `models/ocread_rec/`; `model_dir` must point at the folder that
directly contains the `inference.*` files - with the zip from the notebook
that is `models/ocread_rec/ocread_rec_model`:

    ocr:
      ocread_recognition:
        model_dir: models/ocread_rec/ocread_rec_model
        model_name: PP-OCRv5_mobile_rec

When retraining, replace the files in that folder with the new ones.

Restart the app; the log shows "OCRead uses the recognizer in ...".

## 4. Test it honestly

The model has never seen your four bottles. Per bottle:

    python -m QC_tools.compare_ocr_preprocessing --since 10:54:14 --until 10:56:17 --expected "4113 08:26 BB 05/26"

Compare with the results before fine-tuning (4113: 0 correct, 3030: 0
correct, 5203: 10/10). Keep 5203 at 10/10 while 4113 and 3030 improve.
