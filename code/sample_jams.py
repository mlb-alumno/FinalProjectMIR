import jams
import numpy as np

# Create a new JAMS object
jam = jams.JAMS()

# Add file metadata
jam.file_metadata.artist = 'Artist Name'
jam.file_metadata.title = 'Song Title'
jam.file_metadata.duration = 180.0  # Duration in seconds

# Create a chord annotation
chord_anno = jams.Annotation(namespace='chord')
chord_anno.duration = 180.0  # Duration in seconds

# Add chord intervals and labels
# For example: (start_time, end_time, chord_label)
chords = [
    (0.0, 5.0, 'C:maj'),
    (5.0, 10.0, 'G:maj'),
    (10.0, 15.0, 'A:min'),
    (15.0, 20.0, 'F:maj'),
    # Add more chords...
]

for start, end, chord in chords:
    chord_anno.append(time=start, duration=end-start, value=chord)

# Add annotation to the JAMS object
jam.annotations.append(chord_anno)

# Save to file
jam.save('song_title.jams')
