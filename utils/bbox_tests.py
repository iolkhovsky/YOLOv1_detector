from utils.bbox import BBox

coord = [[20, 40, 10, 15], [200, 200, 14, 13], [5, 1, 200, 150], [0, 0, 224, 224], [120, 130, 50, 75], [190, 190, 34, 34]]

for test in coord:
    print("\n\nSource: ", test)
    test_box = BBox(test[0], test[1], test[2], test[3])
    print("Norm coords: ", test_box.get_norm_coords())
    print("Abs coords: ", test_box.get_abs_coords())
    test_box.rescale(0.3, 2)
    print("Rescaling x = 0.3, y = 2.")
    print("Norm coords: ", test_box.get_norm_coords())
    print("Abs coords: ", test_box.get_abs_coords())


