import os
import sys
import cv2
import cv2.cv2 as cv
import numpy as np


def canny(image):
    return cv2.Canny(image, 100, 200)


def Rinkaku(image, floor):
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img_adth = cv2.adaptiveThreshold(img_gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY,43,3)   #2直化
    img_adth_er = cv2.erode(img_adth, None, iterations = 7)
    img_adth_er = cv2.dilate(img_adth_er, None, iterations = 5) #ノイズ処理
    img_adth_er_re = cv2.bitwise_not(img_adth_er) #白黒反転
    contours, _ = cv2.findContours(img_adth_er_re,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE) #輪郭を抽出
    # 二値化画像(img_adth_er_re)と同じ形状で、0を要素とするndarrayをマスク画像として用意する。
    # (二値化画像と同じ大きさの黒い画像)
    # データ型はopencv用にnp.uint8にする。
    img_mask_adth = np.zeros_like(img_adth, dtype=np.uint8)
    for contour in contours:
    # floorより大きい面積のcontourが対象
        if cv2.contourArea(contour) > floor:
        # 黒色のマスク画像に白色で輪郭を描き、内部を塗りつぶす
        # 引数1=入力画像(要素0のimg_mask)、
#         引数2=輪郭のリスト
#         引数3=描きたい輪郭のindex(-1➡全輪郭を描く)
#         引数4=色(255➡白)
#         引数5=線の太さ(-1➡塗りつぶし)
            cv2.drawContours(img_mask_adth, [contour],-1,255,-1)
    # 黒い部分を確実な背景部とするために、演習⓹でできた画像を膨張させて白い円の領域を大きくする
    sure_bg_adth = cv2.dilate(img_mask_adth,None,iterations=2)
    # 各組織部(白色円)の中心部分からの距離をcv2.distanceTransformで求める
    # distanceTransform(入力画像,距離関数の種類(cv2.DIST_L2=ユークリッド距離),距離変換に用いるマスクの大きさ)
    dist_transform_adth = cv2.distanceTransform(img_mask_adth,cv2.DIST_L2,5)
   # 距離画像(dist_transform_adth)から、確実な前景画像を作成
    # 各組織部の中心からの距離を閾値にして二値化する。
    # 中心部は255で、離れるにつれて色が暗くなるので、今回は最大値255×0.2 = 51を閾値とする。
    ret, sure_fg_adth = cv2.threshold(dist_transform_adth,
                                      0.2*dist_transform_adth.max(),
                                      255, 0)
    # 「確実な背景部」(sure_bg_adth)から「確実な前景部」(sure_fg_adth)を引くことで、unknown領域が得られる。
    sure_fg_adth = np.uint8(sure_fg_adth)
    # cv2.subtractでsure_bg_adthからsure_fg_adthを引く
    unknown_adth = cv2.subtract(sure_bg_adth, sure_fg_adth)
    # 前景の1オブジェクトごとにラベル（番号）を振っていく
    # cv2.connectedComponents() 関数は画像の背景部に0というラベルを与え，それ以外の物体(各前景部)に対して1から順にラベルをつけていく処理をする。．
    ret, markers_adth = cv2.connectedComponents(sure_fg_adth)
    markers_2_adth = markers_adth+1
    # unknownの領域を0にする。
    # unknown_adth画像で白色の部分をmrkers_2_adthの0にする
    markers_2_adth[unknown_adth==255] = 0
    # 次元を合わせるため、演習⓸の画像(img_mask_adth)を3チャンネルにする (元は輪郭を描いただけなので、1チャンネル)
    img_mstack_adth = np.dstack([img_mask_adth]*3)
    # watershedアルゴリズムに適応する
    markers_water_adth = cv2.watershed(img_mstack_adth, markers_2_adth)
    # 境界の領域(-1)を赤で塗る
    image[markers_water_adth == -1] = [0,0,255]

    return image
