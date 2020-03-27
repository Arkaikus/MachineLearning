#Debe existir workspace/data/victimas.csv

##Lectura del archivo que contiene los datos del secuestros
datosTotales = read.csv('./data/victimas.csv',encode="UTF-8")
datosSecuestro=datosTotales[grep("SECUESTRO", datosTotales$DELITO, ignore.case=TRUE),]
secuestroDepartamentos=as.data.frame(sort(table (datosSecuestro$DEPARTAMENTO)))

#Se guardan los secuestros por departamentos
write.csv(secuestroDepartamentos, "./data/SecuestrosPorDepartamentos.csv")

datosDesaparicion = datosTotales[grep("DESAPARICION", datosTotales$DELITO, ignore.case=TRUE),]
desaparicionDepartamentos=as.data.frame(sort(table (datosDesaparicion$DEPARTAMENTO)))
#datosDesaparicion
#desaparicionDepartamentos

#Se guardan las desapariciones por departamentos
write.csv(desaparicionDepartamentos, "./data/DesaparicionPorDepartamentos.csv")

datosAnioDesapariciones= as.data.frame(table(datosDesaparicion$ANIO_HECHO))
#datosAnioDesapariciones

#Se guardan las desapariciones por año
write.csv(datosAnioDesapariciones, "./data/DatosAnioDesapariciones.csv")
