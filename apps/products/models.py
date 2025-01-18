from sqlalchemy import Column, ForeignKey, Integer, String, UniqueConstraint, Text, DateTime, func, Numeric
from sqlalchemy.orm import relationship

from config.database import FastModel


class Product(FastModel):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True)
    product_name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Active: The product is ready to sell and is available to customers on the online store, sales channels, and apps.
    # Archived: The product is no longer being sold and isn't available to customers on sales channels and apps.
    # Draft: The product isn't ready to sell and is unavailable to customers on sales channels and apps.

    # status_enum = Enum('active', 'archived', 'draft', name='status_enum')
    status = Column(String, default='draft')
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, nullable=True)

    # TODO add user_id to track which user added this product


