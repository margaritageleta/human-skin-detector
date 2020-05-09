from SkinDetector import SkinDetector

if __name__ == "__main__":

    sd = SkinDetector()
    sd.train()
    _ = sd.segment(sd.TR_DATA[5], plot = True)

