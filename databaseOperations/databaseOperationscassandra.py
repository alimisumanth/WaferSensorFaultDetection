from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from Utils import  Utils


class Database:
    def __init__(self):
        self.utils = Utils.utils()

    def DBConnection(self):
        cloud_config = {
            'secure_connect_bundle': 'secure-connect-test.zip'
        }
        auth_provider = PlainTextAuthProvider('iRlTdwqURolKnZaQrXstpvGW',
                                              '+QKoyzZqnnZRc-5AZBpxqLn2AUz2AR+o.2YhCLgq00NPpqJl26vBbO0Y-qpODOUutDf-XagT9i4t_kWA3ppIIQIjX0aOBjTP8NNc+7+faYrNqUqDEbWiYgL7_dDQbTBv')
        cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
        session = cluster.connect('test')
        version = session.execute("select release_version from system.local").one()
        print(version)

        return session

    def createTable(self, session):
        config = self.utils.mdm('schema_training.json')
        names = list(config['ColName'].keys())
        names = [''.join(i.replace('-', '_').split()) for i in names]
        dtypes = list(config['ColName'].values())
        res = [i + ' ' + j for i, j in zip(names, dtypes)]
        schema = 'WaferId int,' + ','.join(res)+',primary key(WaferID)'
        print(schema)
        qry = 'create table IF NOT EXISTS Test.students ({}) WITH COMPACT STORAGE;'.format(schema)
        print(qry)
        session.execute(qry)

    def LoadtoDB(self, session):

        query = session.prepare("insert into students (studentID, name, age, marks) values (?,?,?,?);")
        for i in range(10):
            session.execute(query, [i, 'Juhi', i * 10, 200])

    def LoadFromDB(self, session):
        rows = session.execute("select * from students;")
        return rows

    def dropTable(self, session):
        query = 'DROP TABLE IF EXISTS  students'
        session.execute(query)




