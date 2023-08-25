import io

def my_composite(parent_pos, body_type, spacing, count, segement, size, site_size, rgba, damping, stiffness):
    if count <= 1:
        return -1

    with io.open(filename, 'w') as f:
        for i in range(count):
            idx = int(i // (count / segement))
            pos = i * spacing
            geom_txt = "<geom type=\"%s\" pos=\"0 0 %.4f\" size=\"%s\" rgba=\"%s\"/>\n" \
                       % (body_type, pos, size, rgba)
            joint_txt = "<joint type=\"hinge\" pos=\"0 0 %.4f\" axis=\"1 0 0\" damping=\"%.4f\" stiffness=\"%.2f\"/>\n" \
                        "<joint type=\"hinge\" pos=\"0 0 %.4f\" axis=\"0 1 0\" damping=\"%.4f\" stiffness=\"%.2f\"/>\n" \
                        % (pos - spacing / 2, damping, stiffness,
                           pos - spacing / 2, damping, stiffness)
            site_txt = "<site name=\"%s\" pos=\"-%.4f 0 %.4f\"/>\n" \
                       "<site name=\"%s\" pos=\"%.4f 0 %.4f\"/>\n" \
                       "<site name=\"%s\" pos=\"0 -%.4f %.4f\"/>\n" \
                       "<site name=\"%s\" pos=\"0 %.4f %.4f\"/>\n" \
                       % ("s_" + str(idx) + '_1_' + str(i + 1), site_size, pos - spacing / 2,
                          "s_" + str(idx) + '_2_' + str(i + 1), site_size, pos - spacing / 2,
                          "s_" + str(idx) + '_3_' + str(i + 1), site_size, pos - spacing / 2,
                          "s_" + str(idx) + '_4_' + str(i + 1), site_size, pos - spacing / 2)
            meta_txt = "<body>\n" + geom_txt + joint_txt + site_txt
            f.write(meta_txt)

        str_end = ""
        for i in range(count):
            str_end += "</body>\n"
        f.write(str_end)
    return 0

def my_cylinder(site_size, stroke, dist, d_dist, spacing, count, segment):
    with io.open(filename, 'a') as f:
        cpos_a_z = dist + stroke
        cpos_b_z = cpos_a_z + 2 * stroke + d_dist
        cpos_x_y = [[-site_size, 0],
                    [0, -site_size],
                    [site_size, 0],
                    [0, site_size]]

        for i in range(2 * segment):
            cpos_i_x = cpos_x_y[i][0]
            cpos_i_y = cpos_x_y[i][1]
            body_txt = """
            <body>
                <geom type="cylinder" fromto="%(posx).4f %(posy).4f %(posz_a).4f %(posx).4f %(posy).4f %(posz_a_).4f" size="0.01"/>
                <joint name="ctrl_%(idx)d_a" type="slide" axis="0 0 1" limited="true" range="-.1 .1"/>
                <site name="end_%(idx)d_a" pos="%(posx).4f %(posy).4f %(posz_a).4f"/>
            </body>
            <body>
                <geom type="cylinder" fromto="%(posx).4f %(posy).4f %(posz_b).4f %(posx).4f %(posy).4f %(posz_b_).4f" size="0.01"/>
                <joint name="ctrl_%(idx)d_b" type="slide" axis="0 0 1" limited="true" range="-.1 .1"/>
                <site name="end_%(idx)d_b" pos="%(posx).4f %(posy).4f %(posz_b).4f"/>
            </body>
            """ % {"idx": i + 1,
                   "posz_a": -cpos_a_z,
                   "posz_a_": -cpos_a_z - 0.01,
                   "posz_b": -cpos_b_z,
                   "posz_b_": -cpos_b_z - 0.01,
                   "posy": cpos_i_y,
                   "posx": cpos_i_x}
            f.write(body_txt + '\n')

        l_a = stroke + dist + ((count + 1) // segment - 1) * spacing
        l_b = l_a + 2 * stroke + d_dist
        l_c = stroke + dist + (count - 1) * spacing
        l_d = l_c + 2 * stroke + d_dist
        d = 2 * site_size
        print(l_a, l_b, l_c, l_d, d)

        site_txt_1 = []
        site_txt_2 = []
        site_txt_3 = []
        site_txt_4 = []
        for i in range(count):
            idx = int(i // (count / segement))
            site_txt_1.append("<site site=\"s_%s\"/>\n" % (str(idx) + '_1_' + str(i + 1)))
            site_txt_2.append("<site site=\"s_%s\"/>\n" % (str(idx) + '_2_' + str(i + 1)))
            site_txt_3.append("<site site=\"s_%s\"/>\n" % (str(idx) + '_3_' + str(i + 1)))
            site_txt_4.append("<site site=\"s_%s\"/>\n" % (str(idx) + '_4_' + str(i + 1)))

        for str_tmp in site_txt_1:
            f.write(str_tmp)
        for str_tmp in site_txt_2:
            f.write(str_tmp)
        for str_tmp in site_txt_3:
            f.write(str_tmp)
        for str_tmp in site_txt_4:
            f.write(str_tmp)
    return 0

if __name__ == '__main__':
    filename = "test.xml"
    parent_pos = [0, 0, 1]
    body_type = "capsule"
    count = 21  # 16
    segement = 2
    spacing = .12  # .04 12mm
    size = ".01 .015"  # ".01 .015"
    site_size = .07
    rgba = ".8 .2 .1 1"
    damping = .5
    stiffness = 5

    # cylinder
    stroke = 0.1  # 10mm
    dist = 0.1  # Distance between continuum & cylinder
    d_dist = 0.1  # Distance between cylinders

    stat1 = my_composite(parent_pos, body_type, spacing, count, segement, size, site_size, rgba, damping, stiffness)
    if stat1 == 0:
        stat2 = my_cylinder(site_size, stroke, dist, d_dist, spacing, count, segement)
