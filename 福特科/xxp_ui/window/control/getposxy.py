from cv2.typing import map_int_and_double


def getposxy(x_start,y_start):
    pos_list = []
    # x_start=-12.1
    # y_start=-29.1

    y = y_start
    for j in range(6):
        if j%2==0:
            for i in range(6):
                x=round(x_start+4.5*i,1)
                pos_list.append((x,y))
                # print(x,y)
            x_start2=x_start+36.5
            for i in range(6):
                x=round(x_start2+4.5*i,1)
                # print(x,y)
                pos_list.append((x, y))
        else:
            for i in range(6):
                x=round(x_start+59-4.5*i,1)
                # print(x,y)
                pos_list.append((x, y))
            x_start3=22.5+x_start
            for i in range(6):
                x=round(x_start3-4.5*i,1)
                # print(x,y)
                pos_list.append((x, y))
        y=y_start+(j+1)*4.5

    y_start=y_start+36.5
    y=y_start
    for j in range(6):
        if j%2==0:
            for i in range(6):
                x=round(x_start+4.5*i,1)
                # print(x,y)
                pos_list.append((x, y))
            x_start2=x_start+36.5
            for i in range(6):
                x=round(x_start2+4.5*i,1)
                # print(x,y)
                pos_list.append((x, y))
        else:
            for i in range(6):
                x=round(x_start+59-4.5*i,1)

                pos_list.append((x, y))
            x_start3=22.5+x_start
            for i in range(6):
                x=round(x_start3-4.5*i,1)

                pos_list.append((x, y))
        y=y_start+(j+1)*4.5

    return pos_list

if __name__ == '__main__':
    xy_pos = getposxy(-12.3, -27.3)
    print(xy_pos)
    # for (i,pos) in enumerate(xy_pos):
    #     y_now = xy_pos[0][1]
    #     if (i+1) % 12 == 0:
    #         y_now = pos[1]
    #         print(y_now)
    #     else:
    #         x = pos[0]
    #     print(x,y_now)

