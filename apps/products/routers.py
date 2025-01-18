from fastapi import APIRouter, status, Form, UploadFile, File, HTTPException, Query, Path, Depends
from fastapi import Request
from fastapi.responses import JSONResponse

from apps.accounts.services.permissions import Permission
from apps.products import schemas
from apps.products.services import ProductService

router = APIRouter(
    prefix="/products"
)


# -----------------------
# --- Product Routers ---
# -----------------------


@router.post(
    '/',
    status_code=status.HTTP_201_CREATED,
    response_model=schemas.CreateProductOut,
    summary='Create a new product',
    description='Create a new product.',
    tags=["Product"],
    dependencies=[Depends(Permission.is_admin)])
async def create_product(request: Request, product: schemas.CreateProductIn):
    return {'product': ProductService(request).create_product(product.model_dump())}


@router.get(
    '/{product_id}',
    status_code=status.HTTP_200_OK,
    response_model=schemas.RetrieveProductOut,
    summary='Retrieve a single product',
    description="Retrieve a single product.",
    tags=["Product"])
async def retrieve_product(request: Request, product_id: int):
    # TODO user can retrieve products with status of (active , archived)
    # TODO fix bug if there are not product in database
    product = ProductService(request).retrieve_product(product_id)
    return {"product": product}


@router.get(
    '/',
    status_code=status.HTTP_200_OK,
    response_model=schemas.ListProductOut,
    summary='Retrieve a list of products',
    description='Retrieve a list of products.',
    tags=["Product"])
async def list_produces(request: Request):
    # TODO permission: admin users (admin, is_admin), none-admin users
    # TODO as none-admin permission, list products that they status is `active`.
    # TODO as none-admin, dont list the product with the status of `archived` and `draft`.
    # TODO only admin can list products with status `draft`.
    products = ProductService(request).list_products()
    if not products:
        return {'products': []}

    return {'products': products}

@router.put(
    '/{product_id}',
    status_code=status.HTTP_200_OK,
    response_model=schemas.UpdateProductOut,
    summary='Updates a product',
    description='Updates a product.',
    tags=["Product"],
    dependencies=[Depends(Permission.is_admin)])
async def update_product(request: Request, product_id: int, payload: schemas.UpdateProductIn):
    # TODO permission: only admin
    # TODO update a product with media

    updated_product_data = {}
    payload = payload.model_dump()

    for key, value in payload.items():
        if value is not None:
            updated_product_data[key] = value

    try:
        updated_product = ProductService(request).update_product(product_id, **updated_product_data)
        return {'product': updated_product}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete(
    '/{product_id}',
    status_code=status.HTTP_204_NO_CONTENT,
    summary='Deletes an existing product',
    description='Deletes an existing product.',
    tags=['Product'],
    dependencies=[Depends(Permission.is_admin)])
async def delete_product(product_id: int):
    ProductService.delete_product(product_id)

