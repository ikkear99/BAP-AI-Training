from connect_db import Interactive_db

#connect DB
db = Interactive_db()

# test insert
sql = "INSERT INTO mytable(Stt, name, age, address, phuongtien, Bienso, area) VALUES (%s,%s,%s,%s,%s,%s,%s)"
params = ('1000', 'DIENDT','223', 'Hue', 'xemay', '75H34322', '1')
db.insert(sql, params)

#test delete
params = 2
sql = "DELETE FROM khachhang WHERE stt = " + str(params)
db.delete(sql)

#test update
sql = "UPDATE stt  SET name = %s, author = %s WHERE id = %s "
sql = "UPDATE khachhang SET name = %s WHERE stt = %s"
params = ('DienDepTrai',5)
db.update(sql, params)

#test select
sql = "SELECT * FROM khachhang"
data = db.select(sql)

checkpoind = 'debug'