export CUDA_VISIBLE_DEVICES="1"


#(1, 'model.0.bn.gamma'), (2, 'model.1.bn.gamma'), (3, 'model.2.bn.gamma'), (4, 'model.3.bn.gamma'), (5, 'model.4.bn.gamma'),
# (6, 'model.5.bn.gamma'), (7, 'model.6.bn.gamma'), (8, 'model.7.bn.gamma'), (9, 'model.8.bn.gamma'), (10, 'model.9.bn.gamma'),
# (11, 'model.11.bn.gamma'), (12, 'model.13.bn.gamma'), (13, 'model.14.bn.gamma'), (14, 'model.15.bn.gamma'), (15, 'model.17.bn.gamma'),
# (16, 'model.18.bn.gamma'), (17, 'model.19.bn.gamma'), (18, 'model.20.bn.gamma'), (19, 'model.21.bn.gamma'), (20, 'model.22.bn.gamma'),
# (21, 'model.24.bn.gamma'), (22, 'model.26.bn.gamma'), (23, 'model.27.bn.gamma'), (24, 'model.28.bn.gamma'), (25, 'model.30.bn.gamma'),
# (26, 'model.31.bn.gamma'), (27, 'model.32.bn.gamma'), (28, 'model.33.bn.gamma'), (29, 'model.34.bn.gamma'), (30, 'model.35.bn.gamma'),
# (31, 'model.37.bn.gamma'), (32, 'model.39.bn.gamma'), (33, 'model.40.bn.gamma'), (34, 'model.41.bn.gamma'), (35, 'model.43.bn.gamma'),
# (36, 'model.44.bn.gamma'), (37, 'model.45.bn.gamma'), (38, 'model.46.bn.gamma'), (39, 'model.47.bn.gamma'), (40, 'model.48.bn.gamma'),
# (41, 'model.50.bn.gamma'), (42, 'model.51.cv1.bn.gamma'), (43, 'model.51.cv2.bn.gamma'), (44, 'model.51.cv3.bn.gamma'),
# (45, 'model.51.cv4.bn.gamma'), (46, 'model.51.cv5.bn.gamma'), (47, 'model.51.cv6.bn.gamma'), (48, 'model.51.cv7.bn.gamma'),
# (49, 'model.52.bn.gamma'), (50, 'model.54.bn.gamma'), (51, 'model.56.bn.gamma'), (52, 'model.57.bn.gamma'), (53, 'model.58.bn.gamma'),
# (54, 'model.59.bn.gamma'), (55, 'model.60.bn.gamma'), (56, 'model.61.bn.gamma'), (57, 'model.63.bn.gamma'), (58, 'model.64.bn.gamma'),
# (59, 'model.66.bn.gamma'), (60, 'model.68.bn.gamma'), (61, 'model.69.bn.gamma'), (62, 'model.70.bn.gamma'), (63, 'model.71.bn.gamma'),
# (64, 'model.72.bn.gamma'), (65, 'model.73.bn.gamma'), (66, 'model.75.bn.gamma'), (67, 'model.77.bn.gamma'), (68, 'model.78.bn.gamma'),
# (69, 'model.79.bn.gamma'), (70, 'model.81.bn.gamma'), (71, 'model.82.bn.gamma'), (72, 'model.83.bn.gamma'), (73, 'model.84.bn.gamma'),
# (74, 'model.85.bn.gamma'), (75, 'model.86.bn.gamma'), (76, 'model.88.bn.gamma'), (77, 'model.90.bn.gamma'), (78, 'model.91.bn.gamma'),
# (79, 'model.92.bn.gamma'), (80, 'model.94.bn.gamma'), (81, 'model.95.bn.gamma'), (82, 'model.96.bn.gamma'), (83, 'model.97.bn.gamma'),
# (84, 'model.98.bn.gamma'), (85, 'model.99.bn.gamma'), (86, 'model.101.bn.gamma'), (87, 'model.102.rbr_dense_norm.gamma'),
# (88, 'model.102.rbr_1x1_norm.gamma'), (89, 'model.103.rbr_dense_norm.gamma'), (90, 'model.103.rbr_1x1_norm.gamma'),
# (91, 'model.104.rbr_dense_norm.gamma'), (92, 'model.104.rbr_1x1_norm.gamma'), (93, 'model.105.im.0.implicit'),
# (94, 'model.105.im.1.implicit'), (95, 'model.105.im.2.implicit'), (96, 'model.105.ia.0.implicit'), (97, 'model.105.ia.1.implicit'),
# (98, 'model.105.ia.2.implicit'), (99, 'model.0.conv.weight'), (100, 'model.1.conv.weight'), (101, 'model.2.conv.weight'),
# (102, 'model.3.conv.weight'), (103, 'model.4.conv.weight'), (104, 'model.5.conv.weight'), (105, 'model.6.conv.weight'),
# (106, 'model.7.conv.weight'), (107, 'model.8.conv.weight'), (108, 'model.9.conv.weight'), (109, 'model.11.conv.weight'),
# (110, 'model.13.conv.weight'), (111, 'model.14.conv.weight'), (112, 'model.15.conv.weight'), (113, 'model.17.conv.weight'),
# (114, 'model.18.conv.weight'), (115, 'model.19.conv.weight'), (116, 'model.20.conv.weight'), (117, 'model.21.conv.weight'),
# (118, 'model.22.conv.weight'), (119, 'model.24.conv.weight'), (120, 'model.26.conv.weight'), (121, 'model.27.conv.weight'),
# (122, 'model.28.conv.weight'), (123, 'model.30.conv.weight'), (124, 'model.31.conv.weight'), (125, 'model.32.conv.weight'),
# (126, 'model.33.conv.weight'), (127, 'model.34.conv.weight'), (128, 'model.35.conv.weight'), (129, 'model.37.conv.weight'),
# (130, 'model.39.conv.weight'), (131, 'model.40.conv.weight'), (132, 'model.41.conv.weight'), (133, 'model.43.conv.weight'),
# (134, 'model.44.conv.weight'), (135, 'model.45.conv.weight'), (136, 'model.46.conv.weight'), (137, 'model.47.conv.weight'),
# (138, 'model.48.conv.weight'), (139, 'model.50.conv.weight'), (140, 'model.51.cv1.conv.weight'), (141, 'model.51.cv2.conv.weight'),
# (142, 'model.51.cv3.conv.weight'), (143, 'model.51.cv4.conv.weight'), (144, 'model.51.cv5.conv.weight'), (145, 'model.51.cv6.conv.weight'),
# (146, 'model.51.cv7.conv.weight'), (147, 'model.52.conv.weight'), (148, 'model.54.conv.weight'), (149, 'model.56.conv.weight'),
# (150, 'model.57.conv.weight'), (151, 'model.58.conv.weight'), (152, 'model.59.conv.weight'), (153, 'model.60.conv.weight'),
# (154, 'model.61.conv.weight'), (155, 'model.63.conv.weight'), (156, 'model.64.conv.weight'), (157, 'model.66.conv.weight'),
# (158, 'model.68.conv.weight'), (159, 'model.69.conv.weight'), (160, 'model.70.conv.weight'), (161, 'model.71.conv.weight'),
# (162, 'model.72.conv.weight'), (163, 'model.73.conv.weight'), (164, 'model.75.conv.weight'), (165, 'model.77.conv.weight'),
# (166, 'model.78.conv.weight'), (167, 'model.79.conv.weight'), (168, 'model.81.conv.weight'), (169, 'model.82.conv.weight'),
# (170, 'model.83.conv.weight'), (171, 'model.84.conv.weight'), (172, 'model.85.conv.weight'), (173, 'model.86.conv.weight'),
# (174, 'model.88.conv.weight'), (175, 'model.90.conv.weight'), (176, 'model.91.conv.weight'), (177, 'model.92.conv.weight'),
# (178, 'model.94.conv.weight'), (179, 'model.95.conv.weight'), (180, 'model.96.conv.weight'), (181, 'model.97.conv.weight'),
# (182, 'model.98.conv.weight'), (183, 'model.99.conv.weight'), (184, 'model.101.conv.weight'), (185, 'model.102.rbr_dense_conv.weight'),
# (186, 'model.102.rbr_1x1_conv.weight'), (187, 'model.103.rbr_dense_conv.weight'), (188, 'model.103.rbr_1x1_conv.weight'),
# (189, 'model.104.rbr_dense_conv.weight'), (190, 'model.104.rbr_1x1_conv.weight'), (191, 'model.105.m.0.weight'),
# (192, 'model.105.m.1.weight'), (193, 'model.105.m.2.weight'), (194, 'model.0.bn.beta'), (195, 'model.1.bn.beta'),
# (196, 'model.2.bn.beta'), (197, 'model.3.bn.beta'), (198, 'model.4.bn.beta'), (199, 'model.5.bn.beta'), (200, 'model.6.bn.beta'),
# (201, 'model.7.bn.beta'), (202, 'model.8.bn.beta'), (203, 'model.9.bn.beta'), (204, 'model.11.bn.beta'), (205, 'model.13.bn.beta'),
# (206, 'model.14.bn.beta'), (207, 'model.15.bn.beta'), (208, 'model.17.bn.beta'), (209, 'model.18.bn.beta'), (210, 'model.19.bn.beta'),
# (211, 'model.20.bn.beta'), (212, 'model.21.bn.beta'), (213, 'model.22.bn.beta'), (214, 'model.24.bn.beta'), (215, 'model.26.bn.beta'),
# (216, 'model.27.bn.beta'), (217, 'model.28.bn.beta'), (218, 'model.30.bn.beta'), (219, 'model.31.bn.beta'), (220, 'model.32.bn.beta'),
# (221, 'model.33.bn.beta'), (222, 'model.34.bn.beta'), (223, 'model.35.bn.beta'), (224, 'model.37.bn.beta'), (225, 'model.39.bn.beta'),
# (226, 'model.40.bn.beta'), (227, 'model.41.bn.beta'), (228, 'model.43.bn.beta'), (229, 'model.44.bn.beta'), (230, 'model.45.bn.beta'),
# (231, 'model.46.bn.beta'), (232, 'model.47.bn.beta'), (233, 'model.48.bn.beta'), (234, 'model.50.bn.beta'), (235, 'model.51.cv1.bn.beta'),
# (236, 'model.51.cv2.bn.beta'), (237, 'model.51.cv3.bn.beta'), (238, 'model.51.cv4.bn.beta'), (239, 'model.51.cv5.bn.beta'),
# (240, 'model.51.cv6.bn.beta'), (241, 'model.51.cv7.bn.beta'), (242, 'model.52.bn.beta'), (243, 'model.54.bn.beta'),
# (244, 'model.56.bn.beta'), (245, 'model.57.bn.beta'), (246, 'model.58.bn.beta'), (247, 'model.59.bn.beta'),
# (248, 'model.60.bn.beta'), (249, 'model.61.bn.beta'), (250, 'model.63.bn.beta'), (251, 'model.64.bn.beta'),
# (252, 'model.66.bn.beta'), (253, 'model.68.bn.beta'), (254, 'model.69.bn.beta'), (255, 'model.70.bn.beta'),
# (256, 'model.71.bn.beta'), (257, 'model.72.bn.beta'), (258, 'model.73.bn.beta'), (259, 'model.75.bn.beta'),
# (260, 'model.77.bn.beta'), (261, 'model.78.bn.beta'), (262, 'model.79.bn.beta'), (263, 'model.81.bn.beta'),
# (264, 'model.82.bn.beta'), (265, 'model.83.bn.beta'), (266, 'model.84.bn.beta'), (267, 'model.85.bn.beta'),
# (268, 'model.86.bn.beta'), (269, 'model.88.bn.beta'), (270, 'model.90.bn.beta'), (271, 'model.91.bn.beta'),
# (272, 'model.92.bn.beta'), (273, 'model.94.bn.beta'), (274, 'model.95.bn.beta'), (275, 'model.96.bn.beta'),
# (276, 'model.97.bn.beta'), (277, 'model.98.bn.beta'), (278, 'model.99.bn.beta'), (279, 'model.101.bn.beta'),
# (280, 'model.102.rbr_dense_norm.beta'), (281, 'model.102.rbr_1x1_norm.beta'), (282, 'model.103.rbr_dense_norm.beta'),
# (283, 'model.103.rbr_1x1_norm.beta'), (284, 'model.104.rbr_dense_norm.beta'), (285, 'model.104.rbr_1x1_norm.beta'),
# (286, 'model.105.m.0.bias'), (287, 'model.105.m.1.bias'), (288, 'model.105.m.2.bias')