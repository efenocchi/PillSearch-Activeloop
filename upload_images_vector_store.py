import json
import os
import torch
from torchvision import transforms, models
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from utils import get_pills_info
from global_variable import VECTOR_STORE_PATH_IMAGES_MASKED

PILLS_JSON_FILE_CLEANED = "pills_info_cleaned.json"


model = models.resnet18(pretrained=True)

return_nodes = {"avgpool": "embedding"}
model = create_feature_extractor(model, return_nodes=return_nodes)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


tform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: torch.cat([x, x, x], dim=0) if x.shape[0] == 1 else x
        ),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def embedding_function_images(images, model=model, transform=tform, batch_size=4):
    """Creates a list of embeddings based on a list of image filenames. Images are processed in batches."""

    if isinstance(images, str):
        images = [images]

    # Proceess the embeddings in batches, but return everything as a single list
    embeddings = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack(
            [transform(Image.open(item)) for item in images[i : i + batch_size]]
        )
        batch = batch.to(device)
        with torch.no_grad():
            embeddings += model(batch)["embedding"][:, :, 0, 0].cpu().numpy().tolist()

    return embeddings


def upload_vector_store(vector_store_path: str, data_folder: str, pills_info: dict):
    vector_store = DeepLakeVectorStore(
        path=vector_store_path,
        # runtime={"tensor_db": True},
        overwrite=True,
        tensor_params=[
            {"name": "image", "htype": "image", "sample_compression": "jpg"},
            {"name": "embedding", "htype": "embedding"},
            {"name": "filename", "htype": "text"},
            {"name": "metadata", "htype": "json"},
        ],
    )
    # vector_store = vector_store.vectorstore
    vector_store = vector_store
    # image_path = el
    image_path = [
        os.path.join(
            data_folder, f"{pills_info[el]['image_path'].split('.')[0]}_masked.png"
        )
        for el in pills_info
    ]

    metadata = [
        {"name": el, "pill_text": pills_info[el]["pill_text"]} for el in pills_info
    ]

    vector_store.add(
        image=image_path,
        filename=image_path,
        embedding_function=embedding_function_images,
        embedding_data=image_path,
        metadata=metadata,
    )
    return vector_store


def image_similarity_activeloop(vector_store: DeepLakeVectorStore, image_path: str):
    result = vector_store._vectorstore.search(
        embedding_data=[image_path],
        embedding_function=embedding_function_images,
        k=10,
    )
    return result


if __name__ == "__main__":
    """
    Used to compute the embeddings for the images and to load them into the vector store.
    """
    pills_info = get_pills_info()

    # load the masked images
    data_folder = "output"

    # vector_store = upload_vector_store(vector_store_path, data_folder, pills_info)
    vector_store = DeepLakeVectorStore(
        path=VECTOR_STORE_PATH_IMAGES_MASKED,
    )
    # vector_store = vector_store.vectorstore
    vector_store = vector_store
    image_path = "output/aleve_masked.png"
    image_path = "images/aleve.jpg"

    result = image_similarity_activeloop(vector_store, image_path)

    img = Image.fromarray(result["image"][0])
    img.save(f"results/{image_path.split('/')[1].split('.')[0]}_result.png")

    print(result)
