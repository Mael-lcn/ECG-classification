from TurboDataset import TurboDataset



class TurboDataset_Img(TurboDataset):
    """
    Dataset hérité pour la génération à la volée d'images ECG.
    """
    def __init__(self, data_path, generate_img, h=512, w=512, **kwargs):
        super().__init__(data_path, **kwargs)
        self.h = h
        self.w = w
        self.generate_img = generate_img

    def __iter__(self):
        for batch_x, batch_y, batch_lens in super().__iter__():
            # On passe les signaux ET leurs longueurs valides à la fonction.
            # Elle se débrouille pour ne pas calculer de vide avec OpenCV,
            # et elle renvoie le tenseur d'images + le compte d'images valides.
            batch_images, num_windows = self.generate_img(
                batch_x, 
                batch_lens, 
                h=self.h, 
                w=self.w
            )

            # yield final
            yield batch_images, batch_y, num_windows
