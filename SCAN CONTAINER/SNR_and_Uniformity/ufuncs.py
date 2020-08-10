import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import filters
from skimage.measure import label, regionprops
import pandas as pd
import sfuncs as sf


def f_uni_meta(dicomfile):
    """ extract metadata for slice postion info calculations
    dicomfile = pydicom.dataset.FileDataset"""

    # per-frame functional group sequence
    elem = dicomfile[0x5200, 0x9230]  # Per-frame Functional Groups Sequence
    seq = elem.value  # pydicom.sequence.Sequence
    elem3 = seq[0]  # first frame
    elem4 = elem3.PixelMeasuresSequence  # pydicom.sequence.Sequence

    for xx in elem4:
        pixels_space = xx.PixelSpacing

    return pixels_space


def draw_centre_ROI(bin_mask, img, caseT, imagepath=None):
    # ACCOUNT FOR MISSING SIGNAL AT TOP OF PHANTOM (TRANSVERSE VIEW ONLY).
    if caseT:
        oi = np.zeros_like(img, dtype=np.uint8)  # creates zero array same dimensions as img
        oi[(img > filters.threshold_otsu(img)) == True] = 1  # Otsu threshold on image
        err = cv2.erode(oi, None, iterations=8)

        idx = np.where(err > 0)
        idx = idx[0]  # rows
        idx = idx[0]  # first row

        new_mask = bin_mask.copy()
        new_mask[0:idx, :] = 0
        mask = new_mask

    else:
        mask = bin_mask

    # get centre of phantom and define ROI from there
    label_img, num = label(mask, connectivity=img.ndim, return_num=True)  # labels the mask

    props = regionprops(label_img, coordinates='rc')  # returns region properties for phantom mask ROI
    phantom_centre = props[0].centroid
    pc_row, pc_col = [int(phantom_centre[0]), int(phantom_centre[1])]

    # show detected regions and lines on marker_im
    marker_im = img.copy()
    marker_im = marker_im.astype('uint8')
    marker_im = cv2.cvtColor(marker_im, cv2.COLOR_GRAY2BGR)  # grayscale to colour

    cv2.line(marker_im, (pc_col + 10, pc_row + 10), (pc_col + 10, pc_row - 10), (0, 0, 255), 1)
    cv2.line(marker_im, (pc_col + 10, pc_row - 10), (pc_col - 10, pc_row - 10), (0, 0, 255), 1)
    cv2.line(marker_im, (pc_col - 10, pc_row - 10), (pc_col - 10, pc_row + 10), (0, 0, 255), 1)
    cv2.line(marker_im, (pc_col - 10, pc_row + 10), (pc_col + 10, pc_row + 10), (0, 0, 255), 1)

    area = ((pc_col + 10) - (pc_col - 10)) * ((pc_row + 10) - (pc_row - 10))

    area_aim = 20 * 20
    if area != area_aim:
        print('Signal ROI area is too large/too small')
        sys.exit()

    cv2.imwrite("{0}centre_ROI_image.png".format(imagepath), marker_im)

    return pc_row, pc_col, marker_im


def get_signal_value(imdata, pc_r, pc_c):
    # signal values corresponding to voxels inside each signal ROI (don't use greyscale image!)
    signal0 = np.mean(imdata[pc_r - 10:pc_r + 10, pc_c - 10:pc_c + 10])

    print('Mean signal (total) =', signal0)

    return signal0


def obtain_uniformity_profile(imdata, src, dst, pc_row, pc_col, dist80, caseH, caseV, imagepath=None):
    # src and dst are tuples of (x, y) i.e. (column, row)
    # draw line profile across centre line of phantom
    outputs = []
    improfile = np.copy(imdata)
    improfile = (improfile / np.max(improfile))  # normalised
    improfile = improfile * 255  # greyscale
    improfile = improfile.astype('uint8')
    improfile = cv2.cvtColor(improfile, cv2.COLOR_GRAY2BGR)  # grayscale to colour

    dims = np.shape(imdata)

    for xx in np.linspace(-4, 5, 10):
        if caseH:  # horizontal lines
            src2 = (src[0], int(src[1] + xx))  # starting point (x, y)
            dst2 = (dst[0], int(dst[1] + xx))  # finish point
            # to get line profile output
            rows = np.repeat(src2[1], dims[0])
            cols = np.linspace(src2[0], dst2[0], dims[1])
        if caseV:  # vertical lines
            src2 = (int(src[0] + xx), int(src[1]))  # starting point
            dst2 = (int(dst[0] + xx), int(dst[1]))  # finish point
            # to get line profile output
            rows = np.linspace(src2[1], dst2[1], dims[0])
            cols = np.repeat(src2[0], dims[1])

        output = imdata[np.array(np.round(rows), dtype=int), np.array(np.round(cols), dtype=int)]
        outputs.append(output)

        if xx == 0:
            improfile = display_profile_line(improfile, src2, dst2, pc_row, pc_col, dist80, caseH, caseV, linecolour=(0, 0, 255))
        else:
            improfile = display_profile_line(improfile, src2, dst2, pc_row, pc_col, dist80, caseH, caseV, linecolour=(255, 0, 0))

        if caseH:
            cv2.imwrite("{0}profile_line_imageH.png".format(imagepath), improfile)
        if caseV:
            cv2.imwrite("{0}profile_line_imageV.png".format(imagepath), improfile)

    mean_output = np.mean(outputs, axis=0)

    # plot profile line outputs + mean output vs. voxels sampled
    plt.figure()
    if dist80 != 0:
        plt.subplot(221)
    for ee in range(10):
        plt.plot(outputs[ee], 'b')
    plt.plot(mean_output, 'r')
    plt.xlabel('Voxels')
    plt.ylabel('Signal')
    plt.savefig(imagepath + 'fraction_of_uniformity_profiles.png', orientation='landscape', transparent=True, bbox_inches='tight', pad_inches=0.1)

    return mean_output


def display_profile_line(imdata, src, dst, pc_row, pc_col, dist80, caseH, caseV, linecolour, imagepath=None):
    # display profile line on phantom: from source code of profile_line function
    src_col, src_row = np.asarray(src, dtype=float)  # src = (x, y) = (col, row)
    dst_col, dst_row = np.asarray(dst, dtype=float)

    dims = np.shape(imdata)

    if caseH:
        rows = np.repeat(int(src_row), dims[0])
        cols = np.linspace(int(src_col-1), int(dst_col-1), dims[1])

    if caseV:
        rows = np.linspace(int(src_row-1), int(dst_row-1), dims[0])
        cols = np.repeat(int(src_col), dims[1])

    imdata[np.array(np.round(rows), dtype=int), np.array(np.round(cols), dtype=int)] = linecolour

    # 160 mm regions
    if dist80 != 0:
        if caseH:
            cv2.arrowedLine(imdata, (pc_col, pc_row), (pc_col + dist80, pc_row), (0, 0, 0), thickness=2, tipLength=0.1)
            cv2.arrowedLine(imdata, (pc_col, pc_row), (pc_col - dist80, pc_row), (0, 0, 0), thickness=2, tipLength=0.1)
            cv2.putText(imdata, "160 mm", (pc_col-20, pc_row-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        if caseV:
            cv2.arrowedLine(imdata, (pc_col, pc_row), (pc_col, pc_row + dist80), (0, 0, 0), thickness=2, tipLength=0.1)
            cv2.arrowedLine(imdata, (pc_col, pc_row), (pc_col, pc_row - dist80), (0, 0, 0), thickness=2, tipLength=0.1)
            cv2.putText(imdata, "160 mm", (pc_col + 10, pc_row), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # plot sampled line on phantom to visualise where output comes from
    if imagepath:
        cv2.imwrite("{0}just_another_profile_line_image.png".format(imagepath), imdata)

    return imdata


def uniformity_plots(all_signalH, all_signalV, signalH, signalV, pc_col, pc_row, vlinemaxh, vlinemaxv, dist80mm, uniformity_range, fullimagepath):
    plt.figure(figsize=[20, 6])  # width, height in inches
    plt.subplot(121)
    plt.plot(all_signalH, 'b')
    plt.vlines(pc_col, 0, vlinemaxh, colors='y', linestyles='dashdot')
    plt.vlines(pc_col - dist80mm, 0, vlinemaxh, colors='c', linestyles='dashdot')
    plt.vlines(pc_col + dist80mm, 0, vlinemaxh, colors='m', linestyles='dashdot')
    plt.hlines(uniformity_range[0], 0, len(all_signalH), colors='r', linestyles='dashed')
    plt.hlines(uniformity_range[1], 0, len(all_signalH), colors='r', linestyles='dashed')
    plt.legend(['Signal Profile', 'Centre of Profile', '- 80 mm of Centre', '+ 80 mm of Centre', 'Lower Limit',
                'Upper Limit'], loc='lower left')
    plt.xlabel('Voxels')
    plt.ylabel('Signal')
    plt.title('Horizontal Data')

    plt.subplot(122)
    plt.plot(all_signalV, 'g')
    plt.vlines(pc_row, 0, vlinemaxv, colors='y', linestyles='dashdot')
    plt.vlines(pc_row - dist80mm, 0, vlinemaxv, colors='c', linestyles='dashdot')
    plt.vlines(pc_row + dist80mm, 0, vlinemaxv, colors='m', linestyles='dashdot')
    plt.hlines(uniformity_range[0], 0, len(all_signalV), colors='r', linestyles='dashed')
    plt.hlines(uniformity_range[1], 0, len(all_signalV), colors='r', linestyles='dashed')
    plt.legend(['Signal Profile', 'Centre of Profile', '- 80 mm of Centre', '+ 80 mm of Centre', 'Lower Limit',
                'Upper Limit'], loc='lower left')
    plt.xlabel('Voxels')
    plt.ylabel('Signal')
    plt.title('Vertical Data')
    plt.savefig(fullimagepath + 'uniformity_profiles.png', orientation='landscape', transparent=True,
                bbox_inches='tight', pad_inches=0.1)

    plt.figure()
    plt.plot(signalH, 'b')
    plt.plot(signalV, 'g')
    plt.hlines(uniformity_range[0], 0, len(signalH), colors='r', linestyles='dashed')
    plt.hlines(uniformity_range[1], 0, len(signalH), colors='r', linestyles='dashed')
    plt.legend(['Horizontal Profile', 'Vertical Profile', 'Lower Limit', 'Upper Limit'],
               loc='lower left')
    plt.xlabel('Voxels')
    plt.ylabel('Signal')
    plt.title('Selected Profile for Fractional Uniformity Calculation')
    plt.savefig(fullimagepath + 'fraction_of_uniformity_profiles.png', orientation='landscape', transparent=True,
                bbox_inches='tight', pad_inches=0.1)


def calc_fUniformity(signal, uniformity_range):
    total_no_of_voxels = len(signal)
    no_voxels_in_range = 0
    for dd in range(total_no_of_voxels):
        if uniformity_range[0] <= signal[dd] <= uniformity_range[1]:
            no_voxels_in_range = no_voxels_in_range + 1

    fractional_uniformity = no_voxels_in_range / total_no_of_voxels
    mean_signal = np.mean(signal)
    std_signal = np.std(signal)

    return fractional_uniformity, mean_signal, std_signal


def uniformity_analysis(dcmfile, imdata, img, outpath, cT, cS, cC):
    # dcmfile = full dicom file from pydicom dcmread
    # imdata = raw image data
    # img = greyscale image
    # outpath = path where to save png outputs for results
    # cT, cS, cC: tra, sag, cor

    pixel_dims = f_uni_meta(dcmfile)
    print(pixel_dims)

    # mask phantom and background
    mask, bin_mask = sf.create_2D_mask(img, imagepath=outpath)  # boolean and binary masks

    # draw centre ROI
    pc_row, pc_col, marker_im = draw_centre_ROI(bin_mask, img, cT, imagepath=outpath)

    # get mean signal value in ROI
    mean_signal = get_signal_value(imdata, pc_row, pc_col)

    # define uniformity range
    uniformity_range = [mean_signal - (0.1 * mean_signal), mean_signal + (0.1 * mean_signal)]
    print('Expected signal range =', uniformity_range)

    # Obtain Uniformity Profile(s)
    dims = np.shape(imdata)
    """ define 160 mm region for calculation """
    # +/- 80 mm from centre voxel
    dist80mm = int(np.round(80 / pixel_dims[0]))  # how many voxels travserse 80 mm on image
    """plot horizontal profile"""
    srcH = (0, pc_row)  # LHS starting point (x, y) == (col, row)
    dstH = (dims[1] - 1, pc_row)  # RHS finish point
    all_signalH = obtain_uniformity_profile(imdata, srcH, dstH, pc_row, pc_col, dist80mm, caseH=True,
                                               caseV=False, imagepath=outpath)
    """ plot vertical profile """
    srcV = (pc_col, 0)  # starting point
    dstV = (pc_col, dims[0] - 1)  # finish point
    all_signalV = obtain_uniformity_profile(imdata, srcV, dstV, pc_row, pc_col, dist80mm, caseH=False,
                                               caseV=True, imagepath=outpath)

    """get 160 mm of horizontal profile"""
    signalH = all_signalH[pc_col - dist80mm:pc_col + dist80mm]

    """get 160 mm of vertical profile """
    signalV = all_signalV[pc_row - dist80mm:pc_row + dist80mm]

    # Check length of signal H and signal V is = 160 mm
    if 161 < (len(signalH) * pixel_dims[0]) < 159:
        ValueError('Length of Profile is not 160 mm as specified in MagNET protocol.')
    if 161 < (len(signalV) * pixel_dims[0]) < 159:
        ValueError('Length of Profile is not 160 mm as specified in MagNET protocol.')

    vlinemaxh = [np.max(all_signalH), uniformity_range[1]]
    vlinemaxh = np.max(vlinemaxh)

    vlinemaxv = [np.max(all_signalV), uniformity_range[1]]
    vlinemaxv = np.max(vlinemaxv)

    uniformity_plots(all_signalH, all_signalV, signalH, signalV, pc_col, pc_row, vlinemaxh, vlinemaxv,
                        dist80mm, uniformity_range, outpath)

    # fractional uniformity calculation
    fractional_uniformityH, meanH, stdH = calc_fUniformity(signalH, uniformity_range)
    fractional_uniformityV, meanV, stdV = calc_fUniformity(signalV, uniformity_range)

    if cT:
        print('Fractional X Uniformity = ', fractional_uniformityH, '(mean =', meanH.round(2), 'std. dev. =',
              stdH.round(2), ')')
        print('Fractional Y Uniformity = ', fractional_uniformityV, '(mean =', meanV.round(2), 'std. dev. =',
              stdV.round(2), ')')
        # RESULTS TO EXPORT
        print('__._AUTOMATED RESULTS_.__')
        # create Pandas data frame with auto results
        auto_data = {'Signal Range': uniformity_range,
                     'Fractional Uniformity X': fractional_uniformityH, 'Mean Signal X': meanH.round(2), 'StDev Signal X': stdH.round(2),
                     'Fractional Uniformity Y': fractional_uniformityV, 'Mean Signal Y': meanV.round(2), 'StDev Signal Y': stdV.round(2)}
        auto_df = pd.Series(auto_data)  # columns=['Fractional Uniformity X', 'Fractional Uniformity Y', 'Mean Signal X', 'Mean Signal Y', 'StDev Signal X', 'StDev Signal Y'])
        auto_df = auto_df.to_frame()
        print(auto_df.head())


    if cS:
        print('Fractional Y Uniformity = ', fractional_uniformityH, '(mean =', meanH.round(2), 'std. dev. =',
              stdH.round(2), ')')
        print('Fractional Z Uniformity = ', fractional_uniformityV, '(mean =', meanV.round(2), 'std. dev. =',
              stdV.round(2), ')')
        # RESULTS TO EXPORT
        print('__._AUTOMATED RESULTS_.__')
        # create Pandas data frame with auto results
        auto_data = {'Signal Range': uniformity_range,
                     'Fractional Uniformity Y': fractional_uniformityH, 'Mean Signal Y': meanH.round(2),
                     'StDev Signal Y': stdH.round(2),
                     'Fractional Uniformity Z': fractional_uniformityV, 'Mean Signal Z': meanV.round(2),
                     'StDev Signal Z': stdV.round(2)}
        auto_df = pd.Series(auto_data)  # columns=['Fractional Uniformity Y', 'Fractional Uniformity Z', 'Mean Signal Y', 'Mean Signal Z', 'StDev Signal Y', 'StDev Signal Z'])
        auto_df = auto_df.to_frame()
        print(auto_df.head())

    if cC:
        print('Fractional X Uniformity = ', fractional_uniformityH, '(mean =', meanH.round(2), 'std. dev. =',
              stdH.round(2), ')')
        print('Fractional Z Uniformity = ', fractional_uniformityV, '(mean =', meanV.round(2), 'std. dev. =',
              stdV.round(2), ')')
        # RESULTS TO EXPORT
        print('__._AUTOMATED RESULTS_.__')
        # create Pandas data frame with auto results
        auto_data = {'Signal Range': uniformity_range,
                     'Fractional Uniformity X': fractional_uniformityH, 'Mean Signal X': meanH.round(2),
                     'StDev Signal X': stdH.round(2),
                     'Fractional Uniformity Z': fractional_uniformityV, 'Mean Signal Z': meanV.round(2),
                     'StDev Signal Z': stdV.round(2)}
        auto_df = pd.Series(auto_data)  # columns=['Fractional Uniformity X', 'Fractional Uniformity Z', 'Mean Signal X', 'Mean Signal Z', 'StDev Signal X', 'StDev Signal Z'])
        auto_df = auto_df.to_frame()
        print(auto_df.head())
    auto_df.to_html('{0}uniformity_data.html'.format(outpath))










