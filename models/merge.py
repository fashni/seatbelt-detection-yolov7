import csv
import time
from pathlib import Path
from typing import Callable, Optional


def merge(inputdir: Path, outputdir: Path, filename: str, cleanup: bool = False, callback: Optional[Callable] = None) -> None:
  starttime = time.perf_counter()
  manfile = inputdir / "manifest.csv"
  if not manfile.is_file():
    raise FileNotFoundError("Manifest file does not exist")

  outputfile = outputdir / filename
  with open(manfile, mode='r', encoding='utf8', newline='') as reader:
    with open(outputfile, mode='wb+') as writer:
      csvreader = csv.DictReader(reader)
      skipheader = False
      for line in csvreader:
        splitfilename = line['filename']
        splitfile = inputdir / splitfilename
        header = line['header'].lower() == 'true'
        print(f"Writing file part: {splitfilename}")
        with open(splitfile, mode='rb') as splitreader:
          if skipheader:
            next(splitreader)
          for line in splitreader:
            writer.write(line)
        if header:
          skipheader = True
        if cleanup:
          splitfile.unlink()
  if cleanup:
    manfile.unlink()
  if callback:
    callback(outputfile, outputfile.stat().st_size)

  print(f'Process completed in {time.perf_counter() - starttime:.2f} second(s)')


if __name__ == "__main__":
  inputdir = Path("seatbelt-v7")
  outputdir = Path()
  filename = "seatbelt-v7.onnx"
  merge(inputdir, outputdir, filename)

