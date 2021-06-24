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
                                                   password='',
                                                   db='dienai')
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
        self.__commit()
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

