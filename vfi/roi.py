import cv2

def main():
    img = cv2.imread('./data/ori/000000110.png')
    # 讓你畫出要量 ROI 的區域
    r = cv2.selectROI("Select Center Cards ROI", img, showCrosshair=False, fromCenter=False)
    print("中間牌區 ROI:", r)  # 輸出 (x, y, w, h)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
