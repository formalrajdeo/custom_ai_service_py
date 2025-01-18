from typing import Annotated, List

from fastapi import Query, UploadFile
from pydantic import BaseModel, constr, field_validator, model_validator

"""
---------------------------------------
--------------- Product ---------------
---------------------------------------
"""


class ProductSchema(BaseModel):
    product_id: int
    product_name: Annotated[str, Query(max_length=255)]
    description: str | None
    status: str | None

    created_at: str
    updated_at: str | None

class CreateProductOut(BaseModel):
    product: ProductSchema

    class Config:
        from_attributes = True


class CreateProductIn(BaseModel):
    product_name: Annotated[str, Query(max_length=255, min_length=1)]
    description: str | None = None
    status: str | None = None
    price: float = 0
    stock: int = 0

    class Config:
        from_attributes = True

    @field_validator('price')
    def validate_price(cls, price):
        if price < 0:
            raise ValueError('Price must be a positive number.')
        return price

    @field_validator('stock')
    def validate_stock(cls, stock):
        if stock < 0:
            raise ValueError('Stock must be a positive number.')
        return stock


class RetrieveProductOut(BaseModel):
    product: ProductSchema


class ListProductIn(BaseModel):
    ...


class ListProductOut(BaseModel):
    products: list[ProductSchema]


class UpdateProductIn(BaseModel):
    product_name: Annotated[str, Query(max_length=255, min_length=1)] | None = None
    description: str | None = None
    status: str | None = None


class UpdateProductOut(BaseModel):
    product: ProductSchema
