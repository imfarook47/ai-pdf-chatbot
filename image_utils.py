import fitz
import os


def extract_images_from_pdf(pdf_file, output_folder="temp_images"):

    os.makedirs(output_folder, exist_ok=True)

    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")

    image_paths = []

    for page_index in range(len(pdf_document)):

        page = pdf_document[page_index]

        image_list = page.get_images(full=True)

        for image_index, img in enumerate(image_list):

            xref = img[0]

            base_image = pdf_document.extract_image(xref)

            image_bytes = base_image["image"]

            image_ext = base_image["ext"]

            image_path = os.path.join(
                output_folder,
                f"page_{page_index+1}_{image_index}.{image_ext}"
            )

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            image_paths.append(image_path)

    return image_paths