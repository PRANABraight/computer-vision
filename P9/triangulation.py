"""
P9: Triangulation 

This script demonstrates triangulation using two images captured from slightly different positions.
Features:
1. Detect and match keypoints using ORB, SIFT, and AKAZE
2. Calculate 3D coordinates using triangulation (OpenCV SFM pipeline)
3. Display matched keypoints and print 3D coordinates

"""

import cv2
import numpy as np
import sys


class TeeOutput:
    """Write output to both console and file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()


def detect_and_match_orb(gray1, gray2, n_features=500):
    """
    Detect and match features using ORB detector.
    ORB uses binary descriptors -> use Hamming distance for matching.
    """
    orb = cv2.ORB_create(nfeatures=n_features)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    print(f"ORB: Found {len(kp1)} keypoints in left image")
    print(f"ORB: Found {len(kp2)} keypoints in right image")
    
    if des1 is None or des2 is None:
        return None, None, None
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    print(f"ORB: Found {len(matches)} matches")
    return kp1, kp2, matches


def detect_and_match_sift(gray1, gray2, ratio_thresh=0.75):
    """
    Detect and match features using SIFT detector.
    SIFT uses floating-point descriptors -> use L2 distance with Lowe's ratio test.
    """
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    print(f"SIFT: Found {len(kp1)} keypoints in left image")
    print(f"SIFT: Found {len(kp2)} keypoints in right image")
    
    if des1 is None or des2 is None:
        return None, None, None
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    
    print(f"SIFT: Found {len(good_matches)} good matches after ratio test")
    return kp1, kp2, good_matches


def detect_and_match_akaze(gray1, gray2):
    """
    Detect and match features using AKAZE detector.
    AKAZE is an open-source alternative to SURF with scale-invariant properties.
    """
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(gray1, None)
    kp2, des2 = akaze.detectAndCompute(gray2, None)
    
    print(f"AKAZE: Found {len(kp1)} keypoints in left image")
    print(f"AKAZE: Found {len(kp2)} keypoints in right image")
    
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return None, None, None
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    print(f"AKAZE: Found {len(matches)} matches")
    return kp1, kp2, matches


def triangulate_points(kp1, kp2, matches, img_shape, min_matches=8):
    """
    Triangulate 3D points from matched 2D keypoints using OpenCV SFM pipeline.
    
    Steps:
    1. Camera Intrinsic Matrix (K)
    2. Essential Matrix (E) with RANSAC
    3. Pose Recovery (R, t)
    4. Projection Matrices (P1, P2)
    5. Triangulation
    """
    if matches is None or len(matches) < min_matches:
        print(f"Not enough matches ({len(matches) if matches else 0}) for triangulation.")
        return None, None, None, None, None
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # Camera Intrinsic Matrix (assuming uncalibrated camera)
    h, w = img_shape[:2]
    f = w  # Approximate focal length as image width
    K = np.array([[f, 0, w/2],
                  [0, f, h/2],
                  [0, 0, 1]], dtype=np.float64)
    
    print(f"\nCamera Matrix K:\n{K}\n")
    
    # Find Essential Matrix
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    print(f"Essential Matrix E:\n{E}\n")
    
    # Filter inliers
    pts1_inliers = pts1[mask.ravel() == 1]
    pts2_inliers = pts2[mask.ravel() == 1]
    print(f"Inliers after RANSAC: {len(pts1_inliers)} / {len(pts1)}")
    
    if len(pts1_inliers) < min_matches:
        print("Not enough inliers for pose recovery.")
        return None, None, None, None, None
    
    # Recover Pose (R and t)
    _, R, t, mask_pose = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)
    print(f"\nRotation Matrix R:\n{R}")
    print(f"\nTranslation Vector t:\n{t.T}\n")
    
    # Filter by pose mask
    pts1_final = pts1_inliers[mask_pose.ravel() > 0]
    pts2_final = pts2_inliers[mask_pose.ravel() > 0]
    print(f"Points after pose recovery: {len(pts1_final)}")
    
    if len(pts1_final) < 4:
        print("Not enough points for triangulation.")
        return None, None, None, None, None
    
    # Construct Projection Matrices
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))
    
    # Triangulate Points
    pts1_norm = pts1_final.T.astype(np.float64)
    pts2_norm = pts2_final.T.astype(np.float64)
    points4D = cv2.triangulatePoints(P1, P2, pts1_norm, pts2_norm)
    
    # Convert from homogeneous to 3D coordinates
    points3D = (points4D[:3] / points4D[3]).T
    
    return points3D, pts1_final, pts2_final, R, t


def print_3d_coordinates(points3D, detector_name, max_points=20):
    """Print 3D coordinates in a formatted table."""
    print(f"\n--- 3D Coordinates of Matched Points ({detector_name}) ---")
    print(f"{'Point ID':<10} {'X':<15} {'Y':<15} {'Z':<15}")
    print("-" * 55)
    for i, pt in enumerate(points3D[:max_points]):
        print(f"{i:<10} {pt[0]:<15.4f} {pt[1]:<15.4f} {pt[2]:<15.4f}")
    if len(points3D) > max_points:
        print(f"... and {len(points3D) - max_points} more points")


def visualize_matches(img1, kp1, img2, kp2, matches, title, output_file=None, max_matches=50):
    """Draw and display/save matched keypoints."""
    good_matches = matches[:max_matches] if matches else []
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    try:
        cv2.imshow(title, img_matches)
    except cv2.error:
        pass
    
    if output_file:
        cv2.imwrite(output_file, img_matches)
        print(f"Saved: {output_file}")
    
    return img_matches


def main():
    # Redirect output to both console and file
    tee = TeeOutput('triangulation_output.txt')
    sys.stdout = tee
    
    print("=" * 60)
    print("P9: TRIANGULATION / TWO-FRAME MOTION")
    print("=" * 60)
    print(f"OpenCV Version: {cv2.__version__}\n")
    
    # Load images
    img_left = cv2.imread('left.jpeg')
    img_right = cv2.imread('right.jpeg')
    
    if img_left is None or img_right is None:
        print("Error: Could not load 'left.jpeg' or 'right.jpeg'.")
        print("Creating dummy images for demonstration...")
        img_left = np.zeros((600, 800, 3), dtype=np.uint8)
        img_right = np.zeros((600, 800, 3), dtype=np.uint8)
        for i in range(50):
            pt = (np.random.randint(100, 700), np.random.randint(100, 500))
            cv2.circle(img_left, pt, 5, (255, 255, 255), -1)
            pt2 = (pt[0] + np.random.randint(-20, 20), pt[1] + np.random.randint(-5, 5))
            cv2.circle(img_right, pt2, 5, (255, 255, 255), -1)
    
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    
    print(f"Left Image shape: {img_left.shape}")
    print(f"Right Image shape: {img_right.shape}\n")
    
    # ==================== ORB ====================
    print("=" * 60)
    print("1. ORB FEATURE DETECTION")
    print("=" * 60)
    kp1_orb, kp2_orb, matches_orb = detect_and_match_orb(gray_left, gray_right)
    
    if matches_orb:
        visualize_matches(img_left, kp1_orb, img_right, kp2_orb, matches_orb,
                         "ORB Matches", "orb_matches.jpg")
        
        print("\n" + "-" * 40)
        print("TRIANGULATION WITH ORB")
        print("-" * 40)
        points3D_orb, _, _, _, _ = triangulate_points(
            kp1_orb, kp2_orb, matches_orb, img_left.shape
        )
        if points3D_orb is not None:
            print_3d_coordinates(points3D_orb, "ORB")
            np.savetxt('3d_coordinates_orb.csv', points3D_orb, delimiter=',',
                      header='X,Y,Z', comments='', fmt='%.6f')
            print("Saved: 3d_coordinates_orb.csv")
    
    # ==================== SIFT ====================
    print("\n" + "=" * 60)
    print("2. SIFT FEATURE DETECTION")
    print("=" * 60)
    kp1_sift, kp2_sift, matches_sift = detect_and_match_sift(gray_left, gray_right)
    
    if matches_sift:
        visualize_matches(img_left, kp1_sift, img_right, kp2_sift, matches_sift,
                         "SIFT Matches", "sift_matches.jpg")
        
        print("\n" + "-" * 40)
        print("TRIANGULATION WITH SIFT")
        print("-" * 40)
        points3D_sift, _, _, _, _ = triangulate_points(
            kp1_sift, kp2_sift, matches_sift, img_left.shape
        )
        if points3D_sift is not None:
            print_3d_coordinates(points3D_sift, "SIFT")
            np.savetxt('3d_coordinates_sift.csv', points3D_sift, delimiter=',',
                      header='X,Y,Z', comments='', fmt='%.6f')
            print("Saved: 3d_coordinates_sift.csv")
    
    # ==================== AKAZE ====================
    print("\n" + "=" * 60)
    print("3. AKAZE FEATURE DETECTION (SURF Alternative)")
    print("=" * 60)
    kp1_akaze, kp2_akaze, matches_akaze = detect_and_match_akaze(gray_left, gray_right)
    
    if matches_akaze:
        visualize_matches(img_left, kp1_akaze, img_right, kp2_akaze, matches_akaze,
                         "AKAZE Matches", "akaze_matches.jpg")
        
        print("\n" + "-" * 40)
        print("TRIANGULATION WITH AKAZE")
        print("-" * 40)
        points3D_akaze, _, _, _, _ = triangulate_points(
            kp1_akaze, kp2_akaze, matches_akaze, img_left.shape
        )
        if points3D_akaze is not None:
            print_3d_coordinates(points3D_akaze, "AKAZE")
            np.savetxt('3d_coordinates_akaze.csv', points3D_akaze, delimiter=',',
                      header='X,Y,Z', comments='', fmt='%.6f')
            print("Saved: 3d_coordinates_akaze.csv")
    
    # ==================== SUMMARY ====================
    print("\n" + "=" * 60)
    print("FEATURE DETECTOR COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Detector':<15} {'Keypoints':<20} {'Matches':<15} {'3D Points':<15}")
    print("-" * 65)
    
    orb_kp = f"{len(kp1_orb)} / {len(kp2_orb)}" if kp1_orb else "N/A"
    orb_matches = len(matches_orb) if matches_orb else "N/A"
    orb_3d = len(points3D_orb) if 'points3D_orb' in dir() and points3D_orb is not None else "N/A"
    print(f"{'ORB':<15} {orb_kp:<20} {str(orb_matches):<15} {str(orb_3d):<15}")
    
    sift_kp = f"{len(kp1_sift)} / {len(kp2_sift)}" if kp1_sift else "N/A"
    sift_matches = len(matches_sift) if matches_sift else "N/A"
    sift_3d = len(points3D_sift) if 'points3D_sift' in dir() and points3D_sift is not None else "N/A"
    print(f"{'SIFT':<15} {sift_kp:<20} {str(sift_matches):<15} {str(sift_3d):<15}")
    
    akaze_kp = f"{len(kp1_akaze)} / {len(kp2_akaze)}" if kp1_akaze else "N/A"
    akaze_matches = len(matches_akaze) if matches_akaze else "N/A"
    akaze_3d = len(points3D_akaze) if 'points3D_akaze' in dir() and points3D_akaze is not None else "N/A"
    print(f"{'AKAZE':<15} {akaze_kp:<20} {str(akaze_matches):<15} {str(akaze_3d):<15}")
    
    # Wait for key press to close windows
    try:
        print("\nPress any key to close all windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        pass
    
    print("\nDone!")
    print("Output saved to: triangulation_output.txt")
    
    # Restore stdout and close file
    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()
