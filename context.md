# FR for the Class


Pipeline
- capture the image
- load the image into memory
- pass the image via face detector algorithm to generate faces
- in the list of faces generated, apply face matching
- generate the names of faces visible in the image


Issue
- similar person in multiple photos
    - reduce it by inter and intra class similarity comparison
- generating GT while correcting labels


Future Scope
- 3d reconstruction of faces