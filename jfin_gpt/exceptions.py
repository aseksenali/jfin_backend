import logging


class CollectionDoesNotExistException(Exception):
    def __init__(self, collection_name: str):
        self.message = f"The collection {collection_name} does not exist"
        super().__init__(self.message)
