from fastapi import Request
from sqlalchemy import select, and_, or_

from apps.core.date_time import DateTime
from apps.products.models import Product
from config import settings
from config.database import DatabaseManager


class ProductService:
    request: Request | None = None
    product = None
    price: int | float
    stock: int

    @classmethod
    def __init__(cls, request: Request | None = None):
        cls.request = request

    @classmethod
    def create_product(cls, data: dict, get_obj: bool = False):

        cls._create_product(data)

        if get_obj:
            return cls.product
        return cls.retrieve_product(cls.product.id)

    @classmethod
    def _create_product(cls, data: dict):
        cls.price = data.pop('price', 0)
        cls.stock = data.pop('stock', 0)

        if 'status' in data:
            # Check if the value is one of the specified values, if not, set it to 'draft'
            valid_statuses = ['active', 'archived', 'draft']
            if data['status'] not in valid_statuses:
                data['status'] = 'draft'

        # create a product
        cls.product = Product.create(**data)

    @classmethod
    def retrieve_product(cls, product_id):
        cls.product = Product.get_or_404(product_id)

        product = {
            'product_id': cls.product.id,
            'product_name': cls.product.product_name,
            'description': cls.product.description,
            'status': cls.product.status,
            'created_at': DateTime.string(cls.product.created_at),
            'updated_at': DateTime.string(cls.product.updated_at)
        }
        return product

    @classmethod
    def update_product(cls, product_id, **kwargs):

        # --- init data ---
        # TODO `updated_at` is autoupdate dont need to code
        kwargs['updated_at'] = DateTime.now()

        # --- update product ---
        Product.update(product_id, **kwargs)
        return cls.retrieve_product(product_id)

    @classmethod
    def list_products(cls, limit: int = 12):
        # - if "default variant" is not set, first variant will be
        # - on list of products, for price, get it from "default variant"
        # - if price or stock of default variant is 0 then select first variant that is not 0
        # - or for price, get it from "less price"
        # do all of them with graphql and let the front devs decide witch query should be run.

        # also can override the list `limit` in settings.py
        if hasattr(settings, 'products_list_limit'):
            limit = settings.products_list_limit

        products_list = []

        with DatabaseManager.session as session:
            products = session.execute(
                select(Product.id).limit(limit)
            )

        for product in products:
            products_list.append(cls.retrieve_product(product.id))

        return products_list

    @staticmethod
    def delete_product(product_id):
        Product.delete(Product.get_or_404(product_id))
