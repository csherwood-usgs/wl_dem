# csherwood@usgs.gov, 2026-10-21, generated with Claude Opus 5.5
# Functions copied as-is from csherwood-usgs/ShoreScan (branch runup_crs), runup/CODES/utils_CIRN.py,
# which follows the CIRN Quantitative Coastal Imaging Toolbox (Bruder and Brodie, 2020).
import numpy as np

def CIRNangles2R(azimuth, tilt, roll):
    """
    Computes a 3x3 rotation matrix R to transform world coordinates to camera coordinates using a ZXZ rotation sequence.

    :param azimuth: (float) Horizontal rotation (radians), positive CW from world Z-axis.
    :param tilt: (float) Up/down tilt (radians), 0 is nadir, +90 is horizon.
    :param roll: (float) Side-to-side tilt (radians), 0 is a horizontal flat camera.
    :return: (np.ndarray) 3x3 rotation matrix.
    """

    ## Section 1: Define Rotation Matrix R
    R = np.zeros((3, 3))

    R[0, 0] = -np.cos(azimuth) * np.cos(roll) - np.sin(azimuth) * np.cos(tilt) * np.sin(roll)
    R[0, 1] = np.cos(roll) * np.sin(azimuth) - np.sin(roll) * np.cos(tilt) * np.cos(azimuth)
    R[0, 2] = -np.sin(roll) * np.sin(tilt)

    R[1, 0] = -np.sin(roll) * np.cos(azimuth) + np.cos(roll) * np.cos(tilt) * np.sin(azimuth)
    R[1, 1] = np.sin(roll) * np.sin(azimuth) + np.cos(roll) * np.cos(tilt) * np.cos(azimuth)
    R[1, 2] = np.cos(roll) * np.sin(tilt)

    R[2, 0] = np.sin(tilt) * np.sin(azimuth)
    R[2, 1] = np.sin(tilt) * np.cos(azimuth)
    R[2, 2] = -np.cos(tilt)

    return R

def undistortUV(Ud, Vd, intrinsics):
    """
    Undistorts distorted UV coordinates using distortion models from the Caltech lens distortion manuals.

    :param Ud: (np.ndarray) Px1 array of distorted U coordinates.
    :param Vd: (np.ndarray) Px1 array of distorted V coordinates.
    :param intrinsics: (dict) Dictionary of intrinsic camera parameters.
    :return: tuple(np.ndarray, np.ndarray) Px1 arrays of undistorted U and V coordinates.
    """
    
    ## Section 1: Define Coefficients
    fx, fy, c0U, c0V = intrinsics['fx'], intrinsics['fy'], intrinsics['coU'], intrinsics['coV']
    d1, d2, d3, t1, t2 = intrinsics['d1'], intrinsics['d2'], intrinsics['d3'], intrinsics['t1'], intrinsics['t2']
    
    ## Section 2: Provide first guess for dx, dy, and fr using distorted x,y
    # Calculate Distorted camera coordinates x, y, and r
    xd = (Ud - c0U) / fx
    yd = (Vd - c0V) / fy
    rd = np.sqrt(xd**2 + yd**2)
    r2d = rd**2

    # Calculate First Guess for Aggregate Coefficients
    fr1 = 1 + d1 * r2d + d2 * r2d**2 + d3 * r2d**3
    dx1 = 2 * t1 * xd * yd + t2 * (r2d + 2 * xd**2)
    dy1 = t1 * (r2d + 2 * yd**2) + 2 * t2 * xd * yd

    ## Section 3: Calculate Undistorted X and Y using first guess
    x = (xd - dx1) / fr1
    y = (yd - dy1) / fr1

    ## Section 4: Iterate until the difference for all values is < 0.001%
    while True:
        # Calculate New Coefficients
        rn = np.sqrt(x**2 + y**2)
        r2n = rn**2
        frn = 1 + d1 * r2n + d2 * r2n**2 + d3 * r2n**3
        dxn = 2 * t1 * x * y + t2 * (r2n + 2 * x**2)
        dyn = t1 * (r2n + 2 * y**2) + 2 * t2 * x * y

        # Determine Percent change from previous fr, dx, and dy values
        chk1 = np.abs(100 * (fr1 - frn) / fr1)
        chk2 = np.abs(100 * (dx1 - dxn) / dx1)
        chk3 = np.abs(100 * (dy1 - dyn) / dy1)

        # Check if all percent changes are less than 0.001%
        if np.all(chk1 < 0.001) and np.all(chk2 < 0.001) and np.all(chk3 < 0.001):
            break

        # Update x, y for the next iteration
        x = (xd - dxn) / frn
        y = (yd - dyn) / frn

        # Update coefficients for the next iteration
        fr1, dx1, dy1 = frn, dxn, dyn

    ## Section 5: Convert x and y to U, V
    U = x * fx + c0U
    V = y * fy + c0V

    return U, V

def distortUV(U, V, intrinsics):
    """
    Distorts undistorted UV coordinates using the distortion model.
    
    :param U (ndarray): Px1 array of undistorted U coordinates.
    :param V (ndarray): Px1 array of undistorted V coordinates.
    :param intrinsics (dict): Dictionary containing the intrinsic camera parameters, including focal length, 
        principal point, and distortion coefficients.

    :return: tuple
        - Ud (ndarray): Px1 array of distorted U coordinates.
        - Vd (ndarray): Px1 array of distorted V coordinates.
        - flag (ndarray): Px1 array indicating if the UVd coordinate is valid (1) or not (0).
    """
    
    fx, fy, c0U, c0V = intrinsics['fx'], intrinsics['fy'], intrinsics['coU'], intrinsics['coV']
    d1, d2, d3, t1, t2 = intrinsics['d1'], intrinsics['d2'], intrinsics['d3'], intrinsics['t1'], intrinsics['t2']
    
    x = (U - c0U) / fx
    y = (V - c0V) / fy
    r2 = x**2 + y**2
    fr = 1 + d1 * r2 + d2 * r2**2 + d3 * r2**3
    dx = 2 * t1 * x * y + t2 * (r2 + 2 * x**2)
    dy = t1 * (r2 + 2 * y**2) + 2 * t2 * x * y
    
    xd = x * fr + dx
    yd = y * fr + dy
    
    Ud = xd * fx + c0U
    Vd = yd * fy + c0V
    
    flag = (r2 < 1).astype(int)  # Validity check based on a threshold
    
    return Ud, Vd, flag

def intrinsicsExtrinsics2P(intrinsics, extrinsics):
    """
    Computes a camera projection matrix from camera intrinsics and extrinsics.

    :param intrinsics: (dict) Dictionary of intrinsic camera parameters.
    :param extrinsics: (dict) Dictionary containing [x, y, z, azimuth, tilt, roll] of the camera (azimuth, tilt, roll should be in degrees).
    :return: tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray) Transformation matrices P, K, R, and IC.
    
    Notes:
        - P: 4x4 transformation matrix to convert XYZ coordinates to distorted UV coordinates
        - K: 3x3 intrinsic matrix to convert XYZc coordinates to distorted UV coordinates
        - R: 3x3 rotation matrix to rotate world XYZ to camera coordinates XYZc
        - IC: 4x3 translation matrix to convert world XYZ to camera coordinates XYZc
    """

    ## Section 1: Format IO into K matrix
    fx, fy, c0U, c0V = intrinsics['fx'], intrinsics['fy'], intrinsics['coU'], intrinsics['coV']
    
    K = np.array([
        [-fx,  0,  c0U],
        [  0, -fy,  c0V],
        [  0,   0,    1]
    ])

    ## Section 2: Compute Rotation Matrix R using CIRN defined angles
    x, y, z = extrinsics['x'], extrinsics['y'], extrinsics['z']
    azimuth, tilt, roll = np.radians([extrinsics['azimuth'], extrinsics['tilt'], extrinsics['roll']])
    R = CIRNangles2R(azimuth, tilt, roll)  # Ensure CIRNangles2R is defined

    ## Section 3: Compute Translation Matrix IC
    IC = np.hstack((np.eye(3), np.array([[-x], [-y], [-z]])))  # 3x4 matrix

    ## Section 4: Compute Camera Projection Matrix P
    P = K @ R @ IC  # Matrix multiplication
    P /= P[2, 3]  # Normalize for homogeneous coordinates

    return P, K, R, IC

def uv_to_xyz(intrinsics, extrinsics, Ud, Vd, known_dim, known_val):
    """
    Converts image coordinates (U, V) to world coordinates (X, Y, Z) using camera intrinsics, extrinsics, and the Direct Linear Transformation (DLT) equations.

    :param intrinsics: (dict) Dictionary of intrinsic camera parameters.
    :param extrinsics: (dict) Dictionary of extrinsic parameters [x, y, z, azimuth, tilt, roll].
    :param Ud: (np.ndarray) Array of distorted U image coordinates.
    :param Vd: (np.ndarray) Array of distorted V image coordinates.
    :param known_dim: (str) The known world coordinate dimension ('x', 'y', or 'z').
    :param known_val: (np.ndarray) The known value of the world coordinate.
    :return: (np.ndarray) Nx3 NumPy array of computed world coordinates [X, Y, Z].
    """

    # Step 1: Convert UV to undistorted image coordinates
    U=Ud
    V=Vd

    # Step 2: Compute camera projection matrix P
    P, _, _, _ = intrinsicsExtrinsics2P(intrinsics, extrinsics)

    known_val = np.array(known_val)
    if known_val.ndim == 0:  # If it's a single integer (scalar)
        known_val = np.full(U.shape, known_val)
    
    # Extract DLT coefficients from P matrix
    A, B, C, D = P[0, :4]
    H, J, K, L = P[1, :4]
    E, F, G = P[2, :3]  # Only 3 elements since the 4th is always 1 in homogeneous coordinates

    # Compute intermediate variables
    M, N, O, P_ = (E * U - A), (F * U - B), (G * U - C), (D - U)
    Q, R, S, T = (E * V - H), (F * V - J), (G * V - K), (L - V)

    # Solve for world coordinates based on the known dimension
    if known_dim.lower() == 'x':
        X = known_val
        Y = ((O * Q - S * M) * X + (S * P_ - O * T)) / (S * N - O * R)
        Z = ((N * Q - R * M) * X + (R * P_ - N * T)) / (R * O - N * S)
    elif known_dim.lower() == 'y':
        Y = known_val
        X = ((O * R - S * N) * Y + (S * P_ - O * T)) / (S * M - O * Q)
        Z = ((M * R - Q * N) * Y + (Q * P_ - M * T)) / (Q * O - M * S)
    elif known_dim.lower() == 'z':
        Z = known_val
        X = ((N * S - R * O) * Z + (R * P_ - N * T)) / (R * M - N * Q)
        Y = ((M * S - Q * O) * Z + (Q * P_ - M * T)) / (Q * N - M * R)
    else:
        raise ValueError("known_dim must be 'x', 'y', or 'z'")

    # Ensure consistent output format
    max_size = max(map(np.size,[X,Y,Z]))
    X = np.full(max_size, X) if np.size(X) == 1 else X
    Y = np.full(max_size, Y) if np.size(Y) == 1 else Y
    Z = np.full(max_size, Z) if np.size(Z) == 1 else Z
   
    xyz = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    return xyz

def dist_uv_to_xyz(intrinsics, extrinsics, Ud, Vd, known_dim, known_val, timeout = 60):
    """
    Converts image coordinates (U, V) to world coordinates (X, Y, Z) using camera intrinsics, extrinsics, and the Direct Linear Transformation (DLT) equations.

    :param intrinsics: (dict) Dictionary of intrinsic camera parameters.
    :param extrinsics: (dict) Dictionary of extrinsic parameters [x, y, z, azimuth, tilt, roll].
    :param Ud: (np.ndarray) Array of distorted U image coordinates.
    :param Vd: (np.ndarray) Array of distorted V image coordinates.
    :param known_dim: (str) The known world coordinate dimension ('x', 'y', or 'z').
    :param known_val: (np.ndarray) The known value of the world coordinate.
    :return: (np.ndarray) Nx3 NumPy array of computed world coordinates [X, Y, Z].
    """

    # Step 1: Convert UV to undistorted image coordinates
    U, V = undistortUV(Ud, Vd, intrinsics)

    # Step 2: Compute camera projection matrix P
    P, _, _, _ = intrinsicsExtrinsics2P(intrinsics, extrinsics)

    # Extract DLT coefficients from P matrix
    A, B, C, D = P[0, :4]
    H, J, K, L = P[1, :4]
    E, F, G = P[2, :3]  # Only 3 elements since the 4th is always 1 in homogeneous coordinates

    # Compute intermediate variables
    M, N, O, P_ = (E * U - A), (F * U - B), (G * U - C), (D - U)
    Q, R, S, T = (E * V - H), (F * V - J), (G * V - K), (L - V)

    # Solve for world coordinates based on the known dimension
    if known_dim == 'x':
        X = known_val
        Y = ((O * Q - S * M) * X + (S * P_ - O * T)) / (S * N - O * R)
        Z = ((N * Q - R * M) * X + (R * P_ - N * T)) / (R * O - N * S)
    elif known_dim == 'y':
        Y = known_val
        X = ((O * R - S * N) * Y + (S * P_ - O * T)) / (S * M - O * Q)
        Z = ((M * R - Q * N) * Y + (Q * P_ - M * T)) / (Q * O - M * S)
    elif known_dim == 'z':
        Z = known_val
        X = ((N * S - R * O) * Z + (R * P_ - N * T)) / (R * M - N * Q)
        Y = ((M * S - Q * O) * Z + (Q * P_ - M * T)) / (Q * N - M * R)
    else:
        raise ValueError("known_dim must be 'x', 'y', or 'z'")

    max_size = max(map(np.size,[X,Y,Z]))
    X = np.full(max_size, X) if np.size(X) == 1 else X
    Y = np.full(max_size, Y) if np.size(Y) == 1 else Y
    Z = np.full(max_size, Z) if np.size(Z) == 1 else Z
   
    # Ensure consistent output format
    xyz = np.column_stack((X.ravel(),Y.ravel(), Z.ravel()))

    return xyz

def xyz_to_dist_uv(intrinsics, extrinsics, xyz):
    """
    Computes the distorted UV coordinates (UVd) for a set of world xyz points given camera EO and IO.
    
    :param intrinsics: (dict) Dictionary of intrinsic camera parameters.
    :param extrinsics: (dict) Dictionary of extrinsic parameters [x, y, z, azimuth, tilt, roll].
    :param xyz: (np.ndarray) Nx3 NumPy array of computed world coordinates [X, Y, Z]
    :returns UVd: (np.ndarray) 2Px1 array of distorted UV coordinates.
    """

    # Compute P matrix containing both intrinsics and extrinsics information
    P, K, R, IC = intrinsicsExtrinsics2P(intrinsics, extrinsics)
    
    # Compute undistorted UV coordinates
    xyz_homogeneous = np.hstack((xyz, np.ones((xyz.shape[0], 1))))  # Convert to homogeneous coordinates
    UV = P @ xyz_homogeneous.T
    UV /= UV[2, :]  # Normalize to make homogeneous
    
    # Distort the undistorted UV coordinates
    Ud, Vd, flag = distortUV(UV[0, :], UV[1, :], intrinsics)
    flag = np.ones(Ud.shape[0])
    # Compute camera coordinates and check for negative Z values
    xyzC = R @ IC @ xyz_homogeneous.T
    flag[xyzC[2, :] <= 0] = 0
    
    # Stack Ud and Vd to form a single matrix
    UVd = np.vstack((Ud, Vd))
    
    return UVd, flag