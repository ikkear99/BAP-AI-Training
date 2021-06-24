from distutils.command.config import config

from mysql import connector

class Interactive_db():

    def __init__(self, **kwargs):
        super(Interactive_db, self).__init__()
        self.__params = kwargs
        self.__connect = None
        self.__cursor = None
        self._init_()

    def _init_(self):
        self.__connect_db()
        self.__get_cursor()

    def __connect_db(self):
        """
        Function connect user with database
        :return: connector
        """
        if not self.__connect:
            try:
                self.__connect = connector.connect(host='localhost',
                                                   port='3306',
                                                   user='root',
                                                   password='tranduydien1999',
                                                   db='bap_ai')
            except:
                print("Cant not connection")
            finally:
                print("Connected")
                return self.__connect

    def __get_cursor(self):
        """
        this method is used to create cursor
        :return: cursor
        """
        if not self.__cursor:
            if not self.__connect:
                self.__connect_db()
            self.__cursor = self.__connect.cursor()
        return self.__cursor

    def __execute(self, sql, params: tuple = None):
        """
        This method is used to execute a query
        :param sql: the sql code to execute
        :param params:
        :return:
        """
        if params:
            self.__cursor.execute(sql, params)
        else:
            self.__cursor.execute(sql)

    def __execute_many(self, sql, seq_params):
        """
        This method is used to execute a query that effect on multi records
        :param sql: the sql statement to execute
        :param seq_params: sequence of dataset to work with statement
        :return:
        """
        self.__cursor.executemany(sql, seq_params)

    def __commmit(self):
        self.__connect.commit()

    def __fetchall(self):
        result = self.__cursor.fetchall()
        return result if len(result) else []

    def select(self, sql):
        """
        The method used get all records
        @param sql: the sql code to execute
        @return: list result
        """
        self.__execute(sql)
        result = self.__fetchall()
        return result

    def insert(self, sql, params):
        """
        the method is used to insert new record
        @param sql: the sql code to execute
        @param params: data implement
        @return:
        """
        self.__execute(sql, params)
        self.__commmit()
        print("Insert Successfully")

    def insert_list(self, sql, seq_params):
        """
        This method is used to insert multi records into table
        :param sql: the sql statement to execute
        :param seq_params: sequence of dataset to work with statement
        :return:
        """
        self.__execute_many(sql, seq_params)
        self.__commmit()
        print("Inserted List Successfully")

    def update(self, sql, params):
        """
        This method is used to update record by id
        @param sql: the sql code to execute
        @param params: data implement
        @return:
        """
        self.__execute(sql, params)
        self.__commmit()
        print("Update Successfully")

    def delete(self, sql):
        """
        This method is used to delete record by ID
        @param sql: sql code to execute
        @return:
        """
        self.__execute(sql)
        self.__commmit()
        print("Delete Successfully")

    def __close_connect(self):
        """
        This method is used to close connect
        @return:
        """
        if self.__connect.is_connected():
            if self.__cursor:
                self.__cursor.close()
            self.__connect.close()
            print("Connected closed")
        self.__connect = None
        self.__cursor = None



    # def connect_mySQL(self):
    #     """
    #     connect to mySQL
    #     @return: check connect
    #     """
    #     try:
    #         conn = connector.connect(host=self.__host,
    #                                     user=self.__user,
    #                                     password=self.__password,
    #                                     port=self.__port
    #                                     )
    #         print('connect accept')
    #         return conn
    #     except:
    #         print('Can not connect MySQL')
    #
    # def connect_Database(self):
    #     """
    #     connect to database
    #     @return: check connect
    #     """
    #     try:
    #         conn = connector.connect(host=self.__host,
    #                                     user=self.__user,
    #                                     password=self.__password,
    #                                     port=self.__port,
    #                                     database=self.__database
    #                                     )
    #         print('connect accept')
    #         return conn
    #
    #     except:
    #         print('can not connect Database')
    #
    # def insert(self, sql: str, *params):
    #     """
    #     insert data to mySQL
    #     @param sql: mySQL insert code ('''insert into tableName(attribute) value()''')
    #     @param params: params data to insert
    #     @return: insert data to tableName
    #     """
    #     try:
    #         conn = self.connect_Database()
    #         cur = conn.cursor()
    #         cur.execute(sql, params)
    #         conn.commit()
    #         print('accept')
    #     except:
    #         print('can not insert')
    #
    #
    # def delete(self, sql1):
    #     """
    #     delete data to mySQL
    #     @param sql1: sql delete code ('''delete from tableName where attribute = value()''')
    #     @return: delete data to tableName
    #     """
    #     try:
    #         conn = self.connect_Database()
    #         cur = conn.cursor()
    #         cur.execute(sql1)
    #         conn.commit()
    #         print('accept')
    #     except:
    #         print('can not delete data')
    #
    # def update(self, sql):
    #     """
    #     update Database
    #     @param sql: MySQL update code
    #     @return: update database on MySQL
    #     """
    #     try:
    #         conn = self.connect_Database()
    #         cur = conn.cursor()
    #         cur.execute(sql)
    #         conn.commit()
    #         print('accept')
    #     except:
    #         print('can not update')