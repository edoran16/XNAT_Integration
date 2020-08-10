import sys
import numpy as np
import cv2
from skimage import filters
from skimage.morphology import convex_hull_image, opening
from skimage import exposure as ex
from skimage.measure import label, regionprops
import pandas as pd


def testing_xtra_scripts(var):
    print('The testing_xtra_script function works!')
    x = var
    y = 5
    print(x, y)
    z = np.mean([x, y])
    print(z)

    output = z

    return output


def snr_analysis(dcmfile, imdata, img, outpath, cT, cS, cC):
    # dcmfile = full dicom file from pydicom dcmread
    # imdata = raw image data
    # img = greyscale image
    # outpath = path where to save png outputs for results

    pixel_dims, slice_thickness, no_averages, pixelBW, freqX, TRep, No_PEsteps = snr_meta(dcmfile)
    print(pixel_dims, slice_thickness, no_averages, pixelBW, freqX, TRep, No_PEsteps)

    # mask phantom and background
    mask, bin_mask = create_2D_mask(img, imagepath=outpath)  # boolean and binary masks

    # draw signal ROIs
    pc_row, pc_col, quad_centres, marker_im = draw_signal_ROIs(bin_mask, img, show_bbox=False, show_quad=False, imagepath=outpath)

    # get signal value
    mean_signal, all_signals = get_signal_value(imdata, pc_row, pc_col, quad_centres)
    factor = 0.655  # for single element coil, background noise follows Rayleigh distribution IPEM Report 112

    # draw background ROIs
    bROIs = draw_background_ROIs(mask, marker_im, pc_col, cT, cS, cC, imagepath=outpath)

    # get background/noise value
    b_noise, all_noise = get_background_noise_value(imdata, bROIs)

    # SNR calculation (background method)
    SNR_background = calc_SNR(factor, mean_signal, b_noise)

    # Normalised SNR calculation
    Qfact = 1
    NSNR, BWcorr, PixelCorr, TimeCorr, TotalCorr = calc_NSNR(pixel_dims, slice_thickness, No_PEsteps, TRep, no_averages,
                                                             SNR_background, Qfactor=Qfact)

    # RESULTS TO EXPORT
    print('__._AUTOMATED RESULTS_.__')
    # create Pandas data frame with auto results
    auto_data = {'Signal ROI': [1, 2, 3, 4, 5], 'Signal Mean': np.round(all_signals, 2),
                 'Background ROI': [1, 2, 3, 4, 5], 'Noise SD': np.round(all_noise, 2)}

    auto_data2 = {'Mean Signal': np.round(mean_signal, 2), 'Mean Noise': np.round(b_noise, 2),
                  'SNR': np.round(SNR_background, 2),
                  'Normalised SNR': np.round(NSNR, 2)}

    auto_df = pd.DataFrame(auto_data, columns=['Signal ROI', 'Signal Mean', 'Background ROI', 'Noise SD'])
    auto_df2 = pd.Series(auto_data2)
    auto_df2 = auto_df2.to_frame()

    print(auto_df)
    auto_df.to_html('{0}snr_data.html'.format(outpath))
    print(auto_df2)
    auto_df2.to_html('{0}snr_results.html'.format(outpath))

    auto_constants_data = {'Bandwidth': 38.4, 'Nominal Bandwidth': 30, 'BW Correction': np.round(BWcorr, 2),
                           'Pixel Dimensions (mm)': np.round(pixel_dims, 2), 'Slice width (mm)': np.round(slice_thickness, 2),
                           'Voxel Correction': np.round(PixelCorr, 2), 'Phase Encoding Steps': np.round(No_PEsteps, 2),
                           'TR': TRep, 'NSA': no_averages,
                           'Scan Time Correction': np.round(TimeCorr, 2), 'Q Normalisation': np.round(Qfact, 2),
                           'Total Correction Factor': np.round(TotalCorr, 2)}
    auto_constants_df = pd.Series(auto_constants_data)
    auto_constants_df = auto_constants_df.to_frame()

    print(auto_constants_df)
    auto_constants_df.to_html('{0}snr_normalisation_constants.html'.format(outpath))

    # CONCAT EVERYTHING
    results_df = pd.concat([auto_df, auto_df2], join='outer')
    results_df2 = pd.concat([results_df, auto_constants_df])
    results_df2 = results_df2.fillna('-')
    print(results_df2)
    results_df2.to_html('{0}snr_results_all.html'.format(outpath))
    ##########################################


def snr_meta(dicomfile):
    """ extract metadata for slice postion info calculations
    dicomfile = pydicom.dataset.FileDataset"""

    # per-frame functional group sequence
    elem = dicomfile[0x5200, 0x9230]  # Per-frame Functional Groups Sequence
    seq = elem.value  # pydicom.sequence.Sequence
    elem3 = seq[0]  # first frame
    elem4 = elem3.PixelMeasuresSequence  # pydicom.sequence.Sequence

    for xx in elem4:
        pixels_space = xx.PixelSpacing
        st = xx.SliceThickness

    # MR Averages Sequence
    elem5 = elem3.MRAveragesSequence
    for yy in elem5:
        NSA = yy.NumberOfAverages

    # (5200, 9229)  Shared Functional Groups Sequence
    elem6 = dicomfile[0x5200, 0x9229]
    seq2 = elem6.value
    elem7 = seq2[0]
    # print(elem7)
    elem8 = elem7.MRImagingModifierSequence
    for zz in elem8:
        PxlBW = zz.PixelBandwidth
        Tx_Freq = zz.TransmitterFrequency

    """ (0018, 9112) MR Timing and Related Parameters Sequence """
    elem9 = elem7.MRTimingAndRelatedParametersSequence
    for aa in elem9:
        TR = aa.RepetitionTime

    """ (0018, 9125) MR FOV / Geometry Sequence """
    elem10 = elem7.MRFOVGeometrySequence
    for bb in elem10:
        N_PE = bb[0x0018, 0x9231].value  # MRAcquisitionPhaseEncodingSteps

    return pixels_space, st, NSA, PxlBW, Tx_Freq, TR, N_PE


def create_2D_mask(img, imagepath=None):
    """ input:  img is  greyscale uint8 image data from DICOM
        imagepath = where to save png
        output: ch is 2D mask (also grayscale!!!!)"""

    h = ex.equalize_hist(img)  # histogram equalisation increases contrast of image

    oi = np.zeros_like(img, dtype=np.uint8)  # creates zero array same dimensions as img
    oi[(img > filters.threshold_otsu(img)) == True] = 255  # Otsu threshold on image

    oh = np.zeros_like(img, dtype=np.uint8)  # zero array same dims as img
    oh[(h > filters.threshold_otsu(h)) == True] = 255  # Otsu threshold on hist eq image

    nm = img.shape[0] * img.shape[1]  # total number of voxels in image
    # calculate normalised weights for weighted combination
    w1 = np.sum(oi) / nm
    w2 = np.sum(oh) / nm
    ots = np.zeros_like(img, dtype=np.uint8)  # create final zero array
    new = (w1 * img) + (w2 * h)  # weighted combination of original image and hist eq version
    ots[(new > filters.threshold_otsu(new)) == True] = 255  # Otsu threshold on weighted combination

    eroded_ots = cv2.erode(ots, None, iterations=3)
    dilated_ots = cv2.dilate(eroded_ots, None, iterations=3)

    openhull = opening(dilated_ots)

    conv_hull = convex_hull_image(openhull)

    ch = np.multiply(conv_hull, 1)  # bool --> binary
    ch = ch.astype('uint8') * 255

    bin_ch = (ch / np.max(ch)).astype('uint8')  # binary mask [0, 1]

    cv2.imwrite("{0}mask.png".format(imagepath), ch)

    return ch, bin_ch


def draw_signal_ROIs(bin_mask, img, show_bbox=False, show_quad=False, imagepath=None):
    """ show_quad = False  # show quadrants for determining signal ROIs on marker image
        show_bbox = False  # show bounding box of phantom on marker image """
    # draw signal ROIs
    # get centre of phantom and definte 5 ROIs from there
    label_img, num = label(bin_mask, connectivity=img.ndim, return_num=True)  # labels the mask

    props = regionprops(label_img)  # returns region properties for phantom mask ROI
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
    print('Centre ROI Area =', area)
    area_aim = 20 * 20
    if area != area_aim:
        print('Signal ROI area is too large/too small')
        sys.exit()

    # draw bounding box around phantom
    bbox = props[0].bbox  # (min_row, min_col, max_row, max_col)
    if show_bbox:
        cv2.line(marker_im, (bbox[1], bbox[0]), (bbox[1], bbox[2]), (255, 255, 255), 1)
        cv2.line(marker_im, (bbox[1], bbox[2]), (bbox[3], bbox[2]), (255, 255, 255), 1)
        cv2.line(marker_im, (bbox[3], bbox[2]), (bbox[3], bbox[0]), (255, 255, 255), 1)
        cv2.line(marker_im, (bbox[3], bbox[0]), (bbox[1], bbox[0]), (255, 255, 255), 1)

    """ DEFINE 4 QUADRANTS (COL,ROW) """
    if show_quad:
        # top left
        cv2.line(marker_im, (bbox[1], bbox[0]), (bbox[1], pc_row), (0, 0, 255), 1)
        cv2.line(marker_im, (bbox[1], pc_row), (pc_col, pc_row), (0, 0, 255), 1)
        cv2.line(marker_im, (pc_col, pc_row), (pc_col, bbox[0]), (0, 0, 255), 1)
        cv2.line(marker_im, (pc_col, bbox[0]), (bbox[1], bbox[0]), (0, 0, 255), 1)
        # top right
        cv2.line(marker_im, (bbox[3], bbox[0]), (bbox[3], pc_row), (100, 200, 0), 1)
        cv2.line(marker_im, (bbox[3], pc_row), (pc_col, pc_row), (100, 200, 0), 1)
        cv2.line(marker_im, (pc_col, pc_row), (pc_col, bbox[0]), (100, 200, 0), 1)
        cv2.line(marker_im, (pc_col, bbox[0]), (bbox[3], bbox[0]), (100, 200, 0), 1)
        # bottom left
        cv2.line(marker_im, (bbox[1], bbox[2]), (bbox[1], pc_row), (0, 140, 255), 1)
        cv2.line(marker_im, (bbox[1], pc_row), (pc_col, pc_row), (0, 140, 255), 1)
        cv2.line(marker_im, (pc_col, pc_row), (pc_col, bbox[2]), (0, 140, 255), 1)
        cv2.line(marker_im, (pc_col, bbox[2]), (bbox[1], bbox[2]), (0, 140, 255), 1)
        # bottom right
        cv2.line(marker_im, (bbox[3], bbox[2]), (bbox[3], pc_row), (255, 0, 0), 1)
        cv2.line(marker_im, (bbox[3], pc_row), (pc_col, pc_row), (255, 0, 0), 1)
        cv2.line(marker_im, (pc_col, pc_row), (pc_col, bbox[2]), (255, 0, 0), 1)
        cv2.line(marker_im, (pc_col, bbox[2]), (bbox[3], bbox[2]), (255, 0, 0), 1)

    """ PUT OTHER 4 ROIs IN CENTRE OF EACH QUADRANT """  # bbox (0min_row, 1min_col, 2max_row, 3max_col)
    # centre coords for each quadrant
    centre1 = [int(((pc_row - bbox[0]) / 2) + bbox[0]), int(((pc_col - bbox[1]) / 2) + bbox[1])]
    centre2 = [int(((pc_row - bbox[0]) / 2) + bbox[0]), int(((bbox[3] - pc_col) / 2) + pc_col)]
    centre3 = [int(((bbox[2] - pc_row) / 2) + pc_row), int(((pc_col - bbox[1]) / 2) + bbox[1])]
    centre4 = [int(((bbox[2] - pc_row) / 2) + pc_row), int(((pc_col - bbox[1]) / 2) + pc_col)]

    quad_centres = [centre1, centre2, centre3, centre4]

    # top left
    cv2.line(marker_im, (centre1[1] + 10, centre1[0] + 10), (centre1[1] + 10, centre1[0] - 10), (0, 0, 255), 1)
    cv2.line(marker_im, (centre1[1] + 10, centre1[0] - 10), (centre1[1] - 10, centre1[0] - 10), (0, 0, 255), 1)
    cv2.line(marker_im, (centre1[1] - 10, centre1[0] - 10), (centre1[1] - 10, centre1[0] + 10), (0, 0, 255), 1)
    cv2.line(marker_im, (centre1[1] - 10, centre1[0] + 10), (centre1[1] + 10, centre1[0] + 10), (0, 0, 255), 1)
    # top right
    cv2.line(marker_im, (centre2[1] + 10, centre2[0] + 10), (centre2[1] + 10, centre2[0] - 10), (100, 200, 0), 1)
    cv2.line(marker_im, (centre2[1] + 10, centre2[0] - 10), (centre2[1] - 10, centre2[0] - 10), (100, 200, 0), 1)
    cv2.line(marker_im, (centre2[1] - 10, centre2[0] - 10), (centre2[1] - 10, centre2[0] + 10), (100, 200, 0), 1)
    cv2.line(marker_im, (centre2[1] - 10, centre2[0] + 10), (centre2[1] + 10, centre2[0] + 10), (100, 200, 0), 1)
    # bottom left
    cv2.line(marker_im, (centre3[1] + 10, centre3[0] + 10), (centre3[1] + 10, centre3[0] - 10), (0, 140, 255), 1)
    cv2.line(marker_im, (centre3[1] + 10, centre3[0] - 10), (centre3[1] - 10, centre3[0] - 10), (0, 140, 255), 1)
    cv2.line(marker_im, (centre3[1] - 10, centre3[0] - 10), (centre3[1] - 10, centre3[0] + 10), (0, 140, 255), 1)
    cv2.line(marker_im, (centre3[1] - 10, centre3[0] + 10), (centre3[1] + 10, centre3[0] + 10), (0, 140, 255), 1)
    # bottom right
    cv2.line(marker_im, (centre4[1] + 10, centre4[0] + 10), (centre4[1] + 10, centre4[0] - 10), (255, 0, 0), 1)
    cv2.line(marker_im, (centre4[1] + 10, centre4[0] - 10), (centre4[1] - 10, centre4[0] - 10), (255, 0, 0), 1)
    cv2.line(marker_im, (centre4[1] - 10, centre4[0] - 10), (centre4[1] - 10, centre4[0] + 10), (255, 0, 0), 1)
    cv2.line(marker_im, (centre4[1] - 10, centre4[0] + 10), (centre4[1] + 10, centre4[0] + 10), (255, 0, 0), 1)

    cv2.imwrite("{0}drawing_signal_rois.png".format(imagepath), marker_im)

    return pc_row, pc_col, quad_centres, marker_im


def get_signal_value(imdata, pc_row, pc_col, quad_centres):
    # signal values corresponding to voxels inside each signal ROI (don't use greyscale image!)
    signal0 = np.mean(imdata[pc_row - 10:pc_row + 10, pc_col - 10:pc_col + 10])

    centre1 = quad_centres[0]
    centre2 = quad_centres[1]
    centre3 = quad_centres[2]
    centre4 = quad_centres[3]

    signal1 = np.mean(imdata[centre1[0] - 10:centre1[0] + 10, centre1[1] - 10:centre1[1] + 10])
    signal2 = np.mean(imdata[centre2[0] - 10:centre2[0] + 10, centre2[1] - 10:centre2[1] + 10])
    signal3 = np.mean(imdata[centre3[0] - 10:centre3[0] + 10, centre3[1] - 10:centre3[1] + 10])
    signal4 = np.mean(imdata[centre4[0] - 10:centre4[0] + 10, centre4[1] - 10:centre4[1] + 10])

    all_signals = [signal0, signal1, signal2, signal3, signal4]

    mean_signal = np.mean(all_signals)  # mean signal from image data (not filtered!)
    print('Mean signal (total) =', mean_signal)

    return mean_signal, all_signals


def check_ROI(roi_mask, phantom_image):
    # phantom_image is binary mask. Need to convert to greyscale.
    if np.max(phantom_image) == 1:  # binary
        phantom_image = phantom_image * 255

    sum_image = roi_mask + phantom_image
    sum_image = sum_image > 255
    sum_sum_image = np.sum(sum_image.astype('uint8'))

    if sum_sum_image > 0:
        print('Error with ROI placement!!! Overlap with phantom.')

    # check ROI area has not extended beyond FOV
    roi_mask = roi_mask / np.max(roi_mask)  # convert to binary mask
    sum_roi_mask = np.sum(roi_mask)

    print('ROI area = ', sum_roi_mask, '(this should be 20 x 20 = 400)')

    if sum_roi_mask != 400:
        print('Error with ROI size! Matrix must extend beyond FOV.')

    if sum_sum_image == 0 and sum_roi_mask == 400:
        print('This ROI is perfectly fine.')


def draw_background_ROIs(mask, marker_im, pc_col, caseT, caseS, caseC, imagepath=None):
    # Background ROIs according to MagNET protocol

    # auto detection of 4 x background ROI samples (one in each corner of background)
    dims = np.shape(mask)
    bin_mask = mask.astype('uint8')

    idx = np.where(mask)  # returns indices where the phantom exists (from Otsu threshold)
    rows = idx[0]
    cols = idx[1]
    min_row = np.min(rows)  # first row of phantom
    max_row = np.max(rows)  # last row of phantom

    min_col = np.min(cols)  # first column of phantom
    max_col = np.max(cols)  # last column of phantom

    mid_row1 = int(round(min_row / 2))
    mid_row2 = int(round((((dims[0] - max_row) / 2) + max_row)))

    bROI1 = np.zeros(np.shape(mask))  # initialise image matrix for each corner ROI
    bROI2 = np.zeros(np.shape(mask))
    bROI3 = np.zeros(np.shape(mask))
    bROI4 = np.zeros(np.shape(mask))
    bROI5 = np.zeros(np.shape(mask))

    # Background ROIs according to MagNET protocol
    if caseT:
        bROI1[mid_row1 - 10:mid_row1 + 10, min_col - 10:min_col + 10] = 255  # top left
        marker_im[mid_row1 - 10:mid_row1 + 10, min_col - 10:min_col + 10] = (0, 0, 255)
        check_ROI(bROI1, bin_mask)

        bROI2[mid_row1 - 10:mid_row1 + 10, max_col - 10:max_col + 10] = 255  # top right
        marker_im[mid_row1 - 10:mid_row1 + 10, max_col - 10:max_col + 10] = (0, 255, 0)
        check_ROI(bROI2, bin_mask)

        bROI3[mid_row2 - 30:mid_row2 - 10, min_col - 10:min_col + 10] = 255  # bottom left
        marker_im[mid_row2 - 30:mid_row2 - 10, min_col - 10:min_col + 10] = (255, 0, 0)
        check_ROI(bROI3, bin_mask)

        bROI4[mid_row2 - 10:mid_row2 + 10, pc_col - 10:pc_col + 10] = 255  # bottom centre
        marker_im[mid_row2 - 10:mid_row2 + 10, pc_col - 10:pc_col + 10] = (0, 140, 255)
        check_ROI(bROI4, bin_mask)

        bROI5[mid_row2 - 30:mid_row2 - 10, max_col - 10:max_col + 10] = 255  # bottom right
        marker_im[mid_row2 - 30:mid_row2 - 10, max_col - 10:max_col + 10] = (205, 235, 255)
        check_ROI(bROI5, bin_mask)

    if caseS or caseC:
        bROI1[mid_row1 - 10:mid_row1 + 10, min_col - 25:min_col - 5] = 255  # top left
        marker_im[mid_row1 - 10:mid_row1 + 10, min_col - 25:min_col - 5] = (0, 0, 255)
        check_ROI(bROI1, bin_mask)

        bROI2[mid_row1 - 10:mid_row1 + 10, max_col + 5:max_col + 25] = 255  # top right
        marker_im[mid_row1 - 10:mid_row1 + 10, max_col + 5:max_col + 25] = (0, 255, 0)
        check_ROI(bROI2, bin_mask)

        bROI3[mid_row2 - 10:mid_row2 + 10, min_col - 25:min_col - 5] = 255  # bottom left
        marker_im[mid_row2 - 10:mid_row2 + 10, min_col - 25:min_col - 5] = (255, 0, 0)
        check_ROI(bROI3, bin_mask)

        bROI4[mid_row2 - 10:mid_row2 + 10, pc_col - 10:pc_col + 10] = 255  # bottom centre
        marker_im[mid_row2 - 10:mid_row2 + 10, pc_col - 10:pc_col + 10] = (0, 140, 255)
        check_ROI(bROI4, bin_mask)

        bROI5[mid_row2 - 10:mid_row2 + 10, max_col + 5:max_col + 25] = 255  # bottom right
        marker_im[mid_row2 - 10:mid_row2 + 10, max_col + 5:max_col + 25] = (205, 235, 255)
        check_ROI(bROI5, bin_mask)

    cv2.imwrite("{0}drawing_bground_rois.png".format(imagepath), marker_im)

    bROIs = [bROI1, bROI2, bROI3, bROI4, bROI5]

    return bROIs


def get_background_noise_value(imdata, bROIs):
    # background/noise voxel values (don't use greyscale image!!)

    bROI1 = bROIs[0]
    bROI2 = bROIs[1]
    bROI3 = bROIs[2]
    bROI4 = bROIs[3]
    bROI5 = bROIs[4]

    n1 = np.std(imdata[np.where(bROI1 == 255)])
    n2 = np.std(imdata[np.where(bROI2 == 255)])
    n3 = np.std(imdata[np.where(bROI3 == 255)])
    n4 = np.std(imdata[np.where(bROI4 == 255)])
    n5 = np.std(imdata[np.where(bROI5 == 255)])

    all_noise = [n1, n2, n3, n4, n5]

    noise = np.mean(all_noise)
    print('Noise in each ROI = ', [n1, n2, n3, n4, n5])
    print('Noise (total) = ', noise)

    return noise, all_noise


def calc_SNR(fact, mean_sig, nse):
    # SNR calculation (background method as opposed to subtraction method)
    SNR_bckgrnd = (fact * mean_sig) / nse
    print('SNR = ', SNR_bckgrnd.round(2))
    return SNR_bckgrnd


def calc_NSNR(pixels_space, st, N_PE, TR, NSA, SNR_background, Qfactor, BW=38.4, BWnom=30):
    # Bandwidth Normalisation
    BWN = np.sqrt(BW) / np.sqrt(BWnom)
    print('Bandwidth normalisation =', BWN.round(2))

    # Voxel Correction - in terms on centimeters to match MagNET Excel report
    # convert from mm to cm
    dx = pixels_space[0] / 10  # ~ 0.09 cm
    dy = pixels_space[1] / 10  # ~ 0.09 cm
    dz = st / 10  # ~ 0.5 cm
    VC = 1 / (dx * dy * dz)
    print('Voxel Correction = ', np.round(VC, 2), 'cm-3')

    # Scan Time Correction - in terms of seconds (not ms)
    STC = 1 / np.sqrt(N_PE * (TR / 1000) * NSA)  # TR in secs
    print('Scan Time Correction = ', STC, 's-1')  # with TR in secs

    # Coil Loading Normalisation
    QN = Qfactor  # depends on test object/coil under investigation
    print('Coil Loading Normalisation = ', QN)

    # Total Correction Factor
    TCF = BWN * VC * STC * QN
    print('Total Correction Factor =', TCF.round(2))

    # Normalised SNR
    NSNR = TCF * SNR_background
    print('Normalised SNR = ', NSNR.round(2))

    return NSNR, BWN, VC, STC, TCF


def dicom_geo(dicomfile):
    # TODO: determine this from the DICOM metadata.
    """ TAGS FOR SIEMENS DATA:
    extract metadata for scan geometry from Series Description and Protcol Name """

    # Series Description
    series_description = dicomfile[0x0008, 0x103e]
    series_description = series_description.value

    # Protocol Name
    protocol_name = dicomfile[0x0018, 0x1030]
    protocol_name = protocol_name.value

    return series_description, protocol_name