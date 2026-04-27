from src import storage


def main() -> None:
    storage.ensure_db()
    result = storage.apply_order_9493_dust_migration()
    print(result)


if __name__ == '__main__':
    main()
