# Indices to KEEP from the 33-point array
# Nose(0), Shoulders(11,12), Elbows(13,14), Wrists(15,16)
# Pinky(17,18), Index(19,20), Thumbs(21,22) <- Keep these for hand rotation!
# Hips(23,24), Knees(25,26), Ankles(27,28)
# Heels(29,30), Foot_Index(31,32) <- Keep these for foot rotation!

keep_indices = [
    0,                  # Nose (Head Anchor)
    7, 8,                # Ears
    11, 12,             # Shoulders
    13, 14,             # Elbows
    15, 16,             # Wrists
    17, 18, 19, 20, 21, 22, # HANDS: Pinky, Index, Thumb (CRITICAL for rotation)
    23, 24,             # Hips
    25, 26,             # Knees
    27, 28,             # Ankles
    29, 30, 31, 32      # FEET: Heel, Toe (CRITICAL for rotation)
]

# Total Input: 25 Points (if keeping thumbs) or 23 points (if dropping indexes)