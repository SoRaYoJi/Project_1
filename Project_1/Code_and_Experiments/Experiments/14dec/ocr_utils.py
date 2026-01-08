import cv2


def sort_boxes(boxes, thresh=30):
    lines = []
    for x,y,w,h in boxes:
        cy = y+h//2
        placed = False
        for line in lines:
            if abs(line[0][4]-cy) < thresh:
                line.append((x,y,w,h,cy)); placed=True; break
        if not placed:
            lines.append([(x,y,w,h,cy)])
    for line in lines:
        line.sort(key=lambda b:b[0])
    return [[b[:4] for b in line] for line in sorted(lines, key=lambda l:l[0][4])]