# Revisión manual de referencias contextualizadas del mitigador

Estas referencias se redactaron con apoyo de OpenAI Codex usando solo el evento canónico y la medida catalogada. El texto candidato de Mistral permaneció oculto durante la redacción.

Cobertura: 111 referencias de 16 tipos de ataque.

Una casilla vacía significa que la revisión del autor sigue pendiente. Marcar la casilla en este documento no modifica por sí solo el estado auditable del JSON.

## Backdoor

Caso: `case-mitlive-01-e15e8e07` · Origen: `edge_iiotset`

- [x] `case-mitlive-01-e15e8e07::mitigation::00`

  - Medida catalogada: aislar el dispositivo con indicios de backdoor de la red operativa
  - Referencia contextualizada: Aislar cautelarmente 192.168.0.128 mientras se valida el segmento TCP saliente con carga hacia 192.168.0.170:4321, sin asumir compromiso a partir de un solo paquete.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-01-e15e8e07::mitigation::01`

  - Medida catalogada: bloquear las comunicaciones salientes y transferencias asociadas con la puerta trasera
  - Referencia contextualizada: Restringir de forma cautelar la comunicación TCP observada de 192.168.0.128:60210 a 192.168.0.170:4321, sin extender el bloqueo a destinos no presentes en la evidencia.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-01-e15e8e07::mitigation::02`

  - Medida catalogada: eliminar la persistencia y reflashear o restaurar el firmware desde una imagen confiable
  - Referencia contextualizada: Examinar la persistencia en 192.168.0.128 y reflashear o restaurar solo si se confirma una alteración, pues el paquete ACK con 208 bytes no demuestra cambios de firmware.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-01-e15e8e07::mitigation::03`

  - Medida catalogada: verificar firma e integridad del firmware antes de cada despliegue
  - Referencia contextualizada: Verificar la firma y la integridad del firmware de 192.168.0.128 antes de desplegarlo de nuevo, como control complementario al análisis del flujo TCP observado.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-01-e15e8e07::mitigation::04`

  - Medida catalogada: aplicar antimalware o listas de aplicaciones permitidas en los equipos compatibles
  - Referencia contextualizada: Aplicar antimalware o listas de aplicaciones permitidas en 192.168.0.128 si el equipo es compatible y revisar qué proceso originó la conexión TCP al puerto 4321.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-01-e15e8e07::mitigation::05`

  - Medida catalogada: capturar muestra de paquetes
  - Referencia contextualizada: Capturar paquetes del flujo 192.168.0.128:60210 a 192.168.0.170:4321 para examinar la carga de 208 bytes y el contexto de la sesión.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-01-e15e8e07::mitigation::06`

  - Medida catalogada: inspeccionar anomalias de protocolo
  - Referencia contextualizada: Inspeccionar el segmento TCP con ACK activo, sin SYN, FIN ni RST, y contrastar su secuencia, acuse y carga con la sesión correspondiente.
  - Estado auditable: `approved_by_author`

## Command_and_Control

Caso: `case-mitlive-16-a9ff1d2b` · Origen: `iot23`

- [x] `case-mitlive-16-a9ff1d2b::mitigation::00`

  - Medida catalogada: aislar el dispositivo que mantiene comunicaciones de mando y control
  - Referencia contextualizada: Aislar 192.168.100.113 solo si la correlación confirma mando y control; el evento muestra un intento TCP fallido de tres paquetes.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-16-a9ff1d2b::mitigation::01`

  - Medida catalogada: bloquear los dominios, direcciones, puertos y protocolos C2 observados
  - Referencia contextualizada: Bloquear, si se valida como indicador C2, la comunicación TCP hacia 128.185.250.50:50; el evento no aporta dominios.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-16-a9ff1d2b::mitigation::02`

  - Medida catalogada: eliminar la persistencia y reflashear el dispositivo desde firmware confiable
  - Referencia contextualizada: Buscar persistencia y valorar el reflasheo solo si el análisis del dispositivo confirma compromiso; el estado S0 no lo demuestra.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-16-a9ff1d2b::mitigation::03`

  - Medida catalogada: rotar credenciales y claves potencialmente expuestas por el canal C2
  - Referencia contextualizada: Rotar credenciales o claves únicamente si aparecen indicios de exposición; el flujo fallido de 180 bytes no los aporta.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-16-a9ff1d2b::mitigation::04`

  - Medida catalogada: aplicar listas permitidas de salida y prevencion de intrusiones para trafico C2
  - Referencia contextualizada: Restringir mediante lista permitida la salida TCP a 128.185.250.50:50 y detectar ese patrón solo si se valida como C2.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-16-a9ff1d2b::mitigation::05`

  - Medida catalogada: segmentar los dispositivos y eliminar servicios y credenciales por defecto
  - Referencia contextualizada: Revisar la segmentación, los servicios y las credenciales por defecto de 192.168.100.113 sin atribuir el flujo a un servicio no identificado.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-16-a9ff1d2b::mitigation::06`

  - Medida catalogada: capturar ventana de flujo para evidencia
  - Referencia contextualizada: Conservar la ventana del flujo TCP saliente de 3,157 segundos, tres paquetes y 180 bytes, incluido su estado S0.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-16-a9ff1d2b::mitigation::07`

  - Medida catalogada: revisar politica de red del segmento
  - Referencia contextualizada: Comprobar si la política del segmento permite conexiones TCP salientes desde 192.168.100.113 hacia 128.185.250.50:50.
  - Estado auditable: `approved_by_author`

## DDoS_HTTP

Caso: `case-mitlive-02-fbe16540` · Origen: `edge_iiotset`

- [x] `case-mitlive-02-fbe16540::mitigation::00`

  - Medida catalogada: aplicar rate limiting de peticiones HTTP por origen, ruta y cliente
  - Referencia contextualizada: Aplicar límites prudentes al tráfico HTTP desde 192.168.0.170 hacia 192.168.0.128:80, sin fijarlos por ruta porque la evidencia no incluye ninguna ruta ni volumen agregado.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-02-fbe16540::mitigation::01`

  - Medida catalogada: activar proteccion DDoS, WAF o proxy inverso delante del servicio HTTP afectado
  - Referencia contextualizada: Proteger con WAF, proxy inverso o mitigación DDoS el servicio HTTP de 192.168.0.128:80 si el análisis confirma presión sobre él, ya que aquí solo consta un paquete.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-02-fbe16540::mitigation::02`

  - Medida catalogada: bloquear temporalmente los origenes persistentes confirmados por la evidencia del flujo
  - Referencia contextualizada: No tratar 192.168.0.170 como origen persistente por este único paquete; bloquearlo temporalmente solo si flujos adicionales confirman recurrencia dañina.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-02-fbe16540::mitigation::03`

  - Medida catalogada: definir limites de conexiones y solicitudes concurrentes en el servicio web
  - Referencia contextualizada: Definir límites de conexiones y solicitudes concurrentes para 192.168.0.128:80 a partir de su línea base; este único paquete ACK es compatible con una conexión ya establecida, pero no permite estimar la concurrencia.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-02-fbe16540::mitigation::04`

  - Medida catalogada: usar cache, balanceo y capacidad upstream para absorber picos de peticiones HTTP
  - Referencia contextualizada: Dimensionar caché, balanceo y capacidad upstream del servicio 192.168.0.128:80 con métricas históricas, porque el evento de 21 bytes no acredita un pico HTTP.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-02-fbe16540::mitigation::05`

  - Medida catalogada: capturar muestra de paquetes
  - Referencia contextualizada: Capturar paquetes alrededor del flujo 192.168.0.170:44538 a 192.168.0.128:80 para identificar la petición HTTP y su secuencia de sesión.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-02-fbe16540::mitigation::06`

  - Medida catalogada: inspeccionar anomalias de protocolo
  - Referencia contextualizada: Inspeccionar el paquete TCP al puerto HTTP con ACK activo, 21 bytes de carga y ausencia de SYN, FIN y RST, y comprobar si pertenece a una sesión ya establecida.
  - Estado auditable: `approved_by_author`

## DDoS_ICMP

Caso: `case-mitlive-03-cc5370e4` · Origen: `edge_iiotset`

- [x] `case-mitlive-03-cc5370e4::mitigation::00`

  - Medida catalogada: limitar la tasa de trafico ICMP en el perimetro y en el segmento afectado
  - Referencia contextualizada: Limitar con prudencia la tasa ICMP de entrada hacia 192.168.0.128 en el perímetro, calibrándola con más tráfico porque la evidencia no aporta volumen.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-03-cc5370e4::mitigation::01`

  - Medida catalogada: filtrar upstream los tipos ICMP no necesarios durante la inundacion
  - Referencia contextualizada: Identificar primero el tipo ICMP observado y solicitar filtrado upstream solo para tipos no necesarios si capturas adicionales confirman una inundación, que este registro sin datos de volumen ni duración no permite establecer.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-03-cc5370e4::mitigation::02`

  - Medida catalogada: bloquear los origenes persistentes y retirar reglas temporales cuando finalice el incidente
  - Referencia contextualizada: No considerar persistente a 153.75.209.162 por un solo paquete; bloquearlo temporalmente únicamente si capturas posteriores confirman recurrencia perjudicial.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-03-cc5370e4::mitigation::03`

  - Medida catalogada: permitir unicamente los tipos y tasas ICMP necesarios para la operacion
  - Referencia contextualizada: Restringir hacia 192.168.0.128 los tipos y tasas ICMP a los necesarios para operar, tras identificar el tipo que la evidencia no especifica.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-03-cc5370e4::mitigation::04`

  - Medida catalogada: establecer alertas de volumen ICMP por dispositivo y segmento
  - Referencia contextualizada: Establecer alertas de volumen ICMP para 192.168.0.128 y su segmento usando una línea base, ya que el registro carece de recuento y duración.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-03-cc5370e4::mitigation::05`

  - Medida catalogada: capturar muestra de paquetes
  - Referencia contextualizada: Capturar paquetes ICMP adyacentes al evento entre 153.75.209.162 y 192.168.0.128 para determinar tipo, frecuencia y respuestas.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-03-cc5370e4::mitigation::06`

  - Medida catalogada: inspeccionar anomalias de protocolo
  - Referencia contextualizada: Inspeccionar el paquete ICMP con secuencia 21809 y checksum 32675 y aclarar la presencia contextual de indicadores ARP y TCP sin inferir un escaneo.
  - Estado auditable: `approved_by_author`

## DDoS_TCP

Caso: `case-mitlive-04-0b1a613a` · Origen: `edge_iiotset`

- [x] `case-mitlive-04-0b1a613a::mitigation::00`

  - Medida catalogada: activar SYN cookies y limitar nuevas conexiones TCP por origen
  - Referencia contextualizada: Activar SYN cookies en 192.168.0.128:80 y limitar nuevas conexiones TCP por origen con umbrales basados en más flujos, pues aquí solo consta un SYN.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-04-0b1a613a::mitigation::01`

  - Medida catalogada: derivar el trafico TCP anomalo a filtrado o scrubbing upstream
  - Referencia contextualizada: Derivar a filtrado upstream el tráfico TCP anómalo hacia 192.168.0.128:80 solo si se confirma un volumen dañino más allá del SYN observado.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-04-0b1a613a::mitigation::02`

  - Medida catalogada: bloquear los origenes persistentes confirmados y eliminar sesiones TCP incompletas
  - Referencia contextualizada: No bloquear 191.242.186.1 por persistencia con este único SYN; revisar la tabla de estado y eliminar únicamente sesiones incompletas confirmadas.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-04-0b1a613a::mitigation::03`

  - Medida catalogada: ajustar colas y tiempos de espera de conexion para evitar agotamiento de recursos
  - Referencia contextualizada: Ajustar colas y tiempos de espera de conexión en 192.168.0.128:80 tras medir sesiones incompletas, dado que el evento no informa de agotamiento.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-04-0b1a613a::mitigation::04`

  - Medida catalogada: monitorizar tasas SYN, conexiones incompletas y reinicios por servicio
  - Referencia contextualizada: Monitorizar en el servicio HTTP de 192.168.0.128:80 la tasa de SYN, las conexiones incompletas y los reinicios, usando este SYN con carga como evento de análisis.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-04-0b1a613a::mitigation::05`

  - Medida catalogada: capturar muestra de paquetes
  - Referencia contextualizada: Capturar paquetes alrededor del flujo 191.242.186.1:30781 a 192.168.0.128:80 para comprobar la negociación y las posibles respuestas.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-04-0b1a613a::mitigation::06`

  - Medida catalogada: inspeccionar anomalias de protocolo
  - Referencia contextualizada: Inspeccionar el SYN TCP dirigido al servicio HTTP en 192.168.0.128:80, con 120 bytes de carga y checksum 13245, dentro del contexto del establecimiento de la conexión TCP.
  - Estado auditable: `approved_by_author`

## DDoS_UDP

Caso: `case-mitlive-05-0790f401` · Origen: `edge_iiotset`

- [x] `case-mitlive-05-0790f401::mitigation::00`

  - Medida catalogada: limitar o descartar el trafico UDP dirigido al servicio afectado
  - Referencia contextualizada: Validar tráfico UDP antes de limitarlo o descartarlo, porque la evidencia disponible solo muestra un SYN TCP desde 192.168.0.128:6476 y no identifica destino.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-05-0790f401::mitigation::01`

  - Medida catalogada: solicitar filtrado anti-spoofing y mitigacion upstream durante la inundacion
  - Referencia contextualizada: Solicitar filtrado anti-spoofing y mitigación upstream solo si se confirma una inundación, ya que el evento observado es un inicio TCP sin respuesta ni destino conocido.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-05-0790f401::mitigation::02`

  - Medida catalogada: bloquear los origenes persistentes y revisar posibles servicios usados para amplificacion
  - Referencia contextualizada: No atribuir persistencia ni amplificación a 192.168.0.128 por este único SYN TCP; revisar servicios UDP solo si nueva evidencia los identifica.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-05-0790f401::mitigation::03`

  - Medida catalogada: cerrar servicios UDP innecesarios y restringir los restantes mediante listas de control
  - Referencia contextualizada: Inventariar los servicios UDP solo después de identificar en capturas adicionales el flujo y el extremo afectados; la evidencia actual describe TCP y carece de destino.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-05-0790f401::mitigation::04`

  - Medida catalogada: establecer baselines y alertas de volumen UDP por puerto y dispositivo
  - Referencia contextualizada: Establecer líneas base y alertas de volumen para el servicio UDP que llegue a identificarse; no usar este evento TCP sin destino ni volumen como referencia UDP.
  - Estado auditable: `approved_by_author`

## DoS

Caso: `case-mitlive-15-32fa99ad` · Origen: `bot_iot`

- [x] `case-mitlive-15-32fa99ad::mitigation::00`

  - Medida catalogada: aislar el servicio o dispositivo afectado y aplicar limites de consumo por origen
  - Referencia contextualizada: Aplicar límites prudentes al origen 192.168.100.148 sobre el servicio UDP de 192.168.100.6:80 y aislarlo solo si se observa impacto operativo.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-15-32fa99ad::mitigation::01`

  - Medida catalogada: activar degradacion controlada, cola o failover para mantener las funciones esenciales
  - Referencia contextualizada: Activar degradación controlada, cola o failover en 192.168.100.6:80 si la monitorización confirma presión de recursos, no demostrada por el flujo de 8 paquetes.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-15-32fa99ad::mitigation::02`

  - Medida catalogada: bloquear la causa confirmada y corregir el fallo que permite agotar recursos
  - Referencia contextualizada: No bloquear 192.168.100.148 como causa confirmada por un solo flujo; correlacionar el estado INT con el servicio y corregir únicamente fallos verificados.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-15-32fa99ad::mitigation::03`

  - Medida catalogada: configurar cuotas, timeouts, watchdogs y mecanismos de backpressure
  - Referencia contextualizada: Configurar cuotas, timeouts, watchdogs y backpressure para el servicio UDP de 192.168.100.6:80 usando métricas de capacidad además de los 8 paquetes observados.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-15-32fa99ad::mitigation::04`

  - Medida catalogada: probar recuperacion y capacidad del servicio frente a agotamiento sostenido
  - Referencia contextualizada: Probar la recuperación y capacidad de 192.168.100.6:80 ante carga UDP sostenida, sin interpretar este flujo de 480 bytes como agotamiento demostrado.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-15-32fa99ad::mitigation::05`

  - Medida catalogada: capturar ventana de flujo para evidencia
  - Referencia contextualizada: Capturar una ventana alrededor del evento 1076895 en ambos sentidos para verificar el estado INT, los 8 paquetes y los 480 bytes registrados.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-15-32fa99ad::mitigation::06`

  - Medida catalogada: revisar politica de red del segmento
  - Referencia contextualizada: Revisar si la política del segmento autoriza UDP desde 192.168.100.148:51647 hacia 192.168.100.6:80 y ajustar solo reglas justificadas.
  - Estado auditable: `approved_by_author`

## Fingerprinting

Caso: `case-mitlive-06-6940c994` · Origen: `edge_iiotset`

- [x] `case-mitlive-06-6940c994::mitigation::00`

  - Medida catalogada: limitar o bloquear el origen que realiza fingerprinting repetitivo
  - Referencia contextualizada: Limitar el tráfico de 192.168.0.128 hacia 192.168.0.170 solo si capturas adicionales confirman un sondeo repetitivo; la evidencia actual describe un paquete.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-06-6940c994::mitigation::01`

  - Medida catalogada: segmentar temporalmente el activo cuya plataforma esta siendo identificada
  - Referencia contextualizada: Segmentar temporalmente 192.168.0.170 solo si se confirma que el tráfico multiprotocolo observado intenta identificar su plataforma.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-06-6940c994::mitigation::02`

  - Medida catalogada: revisar y retirar banners, respuestas y servicios que revelen informacion innecesaria
  - Referencia contextualizada: Revisar en 192.168.0.170 las respuestas asociadas a ARP, ICMP, HTTP y MQTT y retirar únicamente la información innecesaria que se confirme expuesta.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-06-6940c994::mitigation::03`

  - Medida catalogada: reducir la exposicion de versiones y caracteristicas del sistema
  - Referencia contextualizada: Comprobar si 192.168.0.170 expone versiones o características mediante los protocolos observados y, solo si se confirma, reducir esa información; la muestra no aporta esos valores.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-06-6940c994::mitigation::04`

  - Medida catalogada: crear reglas IDS para secuencias de sondeo y huellas conocidas
  - Referencia contextualizada: Crear una regla IDS que correlacione tráfico ARP, ICMP, HTTP y MQTT de 192.168.0.128 hacia 192.168.0.170, sin inferir una secuencia a partir de este paquete.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-06-6940c994::mitigation::05`

  - Medida catalogada: capturar muestra de paquetes
  - Referencia contextualizada: Capturar paquetes adicionales entre 192.168.0.128 y 192.168.0.170 para completar los puertos, el transporte y el volumen ausentes.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-06-6940c994::mitigation::06`

  - Medida catalogada: inspeccionar anomalias de protocolo
  - Referencia contextualizada: Inspeccionar la coherencia de los campos ARP, ICMP, HTTP y MQTT observados, incluidos los valores de aplicación no especificados.
  - Estado auditable: `approved_by_author`

## MITM

Caso: `case-mitlive-07-5e58a71e` · Origen: `ton_iot`

- [x] `case-mitlive-07-5e58a71e::mitigation::00`

  - Medida catalogada: aislar el segmento en el que se observa la intermediacion maliciosa
  - Referencia contextualizada: Aislar el segmento de 192.168.1.34 solo si la revisión del flujo SSL hacia 13.35.146.99:443 confirma intermediación; la evidencia actual no la demuestra.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-07-5e58a71e::mitigation::01`

  - Medida catalogada: forzar canales cifrados y detener temporalmente comunicaciones no autenticadas
  - Referencia contextualizada: Mantener cifrado el flujo de 192.168.1.34 a 13.35.146.99:443 y detenerlo únicamente si se confirma que carece de autenticación.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-07-5e58a71e::mitigation::02`

  - Medida catalogada: eliminar el nodo intruso y corregir entradas ARP o DNS manipuladas
  - Referencia contextualizada: Comprobar los nodos y las tablas ARP o DNS vinculadas a ambos extremos antes de corregirlas, pues la muestra solo acredita un flujo TCP/SSL.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-07-5e58a71e::mitigation::03`

  - Medida catalogada: usar TLS con validacion estricta de certificados para la informacion sensible
  - Referencia contextualizada: Exigir validación estricta de certificados en las conexiones SSL/TLS de 192.168.1.34 hacia 13.35.146.99:443.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-07-5e58a71e::mitigation::04`

  - Medida catalogada: habilitar inspeccion ARP dinamica, DHCP snooping y segmentacion de red
  - Referencia contextualizada: Habilitar inspección ARP dinámica, DHCP snooping y segmentación para la red de 192.168.1.34, sin atribuir manipulación ARP o DHCP a este flujo.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-07-5e58a71e::mitigation::05`

  - Medida catalogada: capturar ventana de flujo para evidencia
  - Referencia contextualizada: Capturar una ventana ampliada del flujo entre 192.168.1.34:49842 y 13.35.146.99:443 para contextualizar sus 50 paquetes y 180,23 segundos.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-07-5e58a71e::mitigation::06`

  - Medida catalogada: revisar politica de red del segmento
  - Referencia contextualizada: Revisar la política de salida del segmento de 192.168.1.34 para conexiones TCP/SSL a destinos externos por el puerto 443.
  - Estado auditable: `approved_by_author`

## Password

Caso: `case-mitlive-08-6da6e3d3` · Origen: `ton_iot`

- [x] `case-mitlive-08-6da6e3d3::mitigation::00`

  - Medida catalogada: aplicar bloqueo temporal o rate limiting tras intentos de autenticacion fallidos
  - Referencia contextualizada: Aplicar bloqueo temporal o rate limiting a 192.168.1.31 solo si registros adicionales confirman intentos de autenticación fallidos; el flujo HTTP observado terminó correctamente.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-08-6da6e3d3::mitigation::01`

  - Medida catalogada: bloquear el origen del ataque contra el servicio de autenticacion
  - Referencia contextualizada: Bloquear 192.168.1.31 frente a 192.168.1.195:80 únicamente si se confirma un ataque contra la autenticación, no demostrado por este flujo.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-08-6da6e3d3::mitigation::02`

  - Medida catalogada: rotar las credenciales afectadas e invalidar las sesiones activas asociadas
  - Referencia contextualizada: Rotar credenciales e invalidar sesiones de 192.168.1.195 solo si registros complementarios identifican credenciales afectadas, dato ausente en la muestra.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-08-6da6e3d3::mitigation::03`

  - Medida catalogada: habilitar autenticacion multifactor donde el dispositivo o servicio lo permita
  - Referencia contextualizada: Habilitar autenticación multifactor en el servicio HTTP de 192.168.1.195 si dispone de acceso autenticado y lo permite; el flujo no muestra ese mecanismo.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-08-6da6e3d3::mitigation::04`

  - Medida catalogada: eliminar credenciales por defecto y aplicar una politica de contrasenas robustas
  - Referencia contextualizada: Revisar y eliminar credenciales por defecto del servicio HTTP de 192.168.1.195 y aplicar una política robusta, sin asumir que este flujo revela su uso.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-08-6da6e3d3::mitigation::05`

  - Medida catalogada: capturar ventana de flujo para evidencia
  - Referencia contextualizada: Capturar una ventana ampliada del flujo entre 192.168.1.31:54554 y 192.168.1.195:80 para obtener datos de autenticación ausentes en la muestra.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-08-6da6e3d3::mitigation::06`

  - Medida catalogada: revisar politica de red del segmento
  - Referencia contextualizada: Revisar la política que permite a 192.168.1.31 acceder a 192.168.1.195 mediante HTTP sobre TCP/80.
  - Estado auditable: `approved_by_author`

## Port_Scanning

Caso: `case-mitlive-09-b0ba0c52` · Origen: `edge_iiotset`

- [x] `case-mitlive-09-b0ba0c52::mitigation::00`

  - Medida catalogada: bloquear o limitar el origen que recorre puertos de forma automatizada
  - Referencia contextualizada: Limitar los SYN de 192.168.0.170 hacia 192.168.0.128 solo si más tráfico confirma un recorrido automatizado de puertos; esta muestra contiene únicamente TCP/80.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-09-b0ba0c52::mitigation::01`

  - Medida catalogada: aislar el segmento escaneado si el reconocimiento precede a intentos de explotacion
  - Referencia contextualizada: Aislar el segmento de 192.168.0.128 únicamente si aparecen intentos de explotación posteriores; el paquete observado solo inicia TCP/80.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-09-b0ba0c52::mitigation::02`

  - Medida catalogada: cerrar puertos y servicios expuestos que no sean necesarios para la operacion
  - Referencia contextualizada: Verificar si TCP/80 en 192.168.0.128 es necesario y cerrarlo solo si no lo exige la operación.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-09-b0ba0c52::mitigation::03`

  - Medida catalogada: aplicar listas de control que permitan solo los puertos y origenes autorizados
  - Referencia contextualizada: Permitir TCP/80 en 192.168.0.128 solo desde orígenes autorizados, revisando específicamente el acceso de 192.168.0.170.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-09-b0ba0c52::mitigation::04`

  - Medida catalogada: configurar IDS o IPS para detectar barridos horizontales y verticales
  - Referencia contextualizada: Configurar IDS o IPS para correlacionar SYN de 192.168.0.170 hacia varios puertos o hosts, sin clasificar este único SYN como barrido.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-09-b0ba0c52::mitigation::05`

  - Medida catalogada: capturar muestra de paquetes
  - Referencia contextualizada: Capturar paquetes adicionales alrededor del SYN de 192.168.0.170:1536 a 192.168.0.128:80 para buscar otros destinos y respuestas.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-09-b0ba0c52::mitigation::06`

  - Medida catalogada: inspeccionar anomalias de protocolo
  - Referencia contextualizada: Inspeccionar el SYN TCP/80 y la ausencia de contenido HTTP antes de atribuir una anomalía de protocolo.
  - Estado auditable: `approved_by_author`

## Ransomware

Caso: `case-mitlive-10-9e17e2c6` · Origen: `ton_iot`

- [x] `case-mitlive-10-9e17e2c6::mitigation::00`

  - Medida catalogada: aislar inmediatamente el equipo afectado para detener la propagacion
  - Referencia contextualizada: Aislar el extremo que resulte afectado entre 192.168.1.37 y 192.168.1.193 únicamente si el análisis adicional confirma actividad de ransomware; la muestra solo contiene un paquete SMB de 60 bytes.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-10-9e17e2c6::mitigation::01`

  - Medida catalogada: desconectar recursos compartidos y almacenamiento accesible desde el equipo
  - Referencia contextualizada: Restringir temporalmente el acceso SMB entre 192.168.1.37 y 192.168.1.193 solo si se confirma riesgo, sin inferir propagación a partir del único paquete.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-10-9e17e2c6::mitigation::02`

  - Medida catalogada: preservar evidencias y reconstruir el sistema desde una imagen limpia verificada
  - Referencia contextualizada: Preservar la muestra y reconstruir un sistema desde una imagen limpia solo si evidencia de host posterior confirma afectación; el flujo no identifica un equipo comprometido.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-10-9e17e2c6::mitigation::03`

  - Medida catalogada: mantener copias de seguridad desconectadas y probar periodicamente su restauracion
  - Referencia contextualizada: Mantener copias desconectadas y probar su restauración para los datos que 192.168.1.193 comparta por SMB, si se confirma que el servicio observado expone almacenamiento.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-10-9e17e2c6::mitigation::04`

  - Medida catalogada: reducir privilegios y mantener actualizados firmware, sistema y aplicaciones
  - Referencia contextualizada: Reducir privilegios de acceso SMB y mantener actualizados 192.168.1.37 y 192.168.1.193, priorizando la revisión del servicio TCP/445 observado.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-10-9e17e2c6::mitigation::05`

  - Medida catalogada: capturar ventana de flujo para evidencia
  - Referencia contextualizada: Capturar una ventana ampliada del flujo de 192.168.1.37:37887 a 192.168.1.193:445 para contextualizar el estado OTH y el paquete de 60 bytes.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-10-9e17e2c6::mitigation::06`

  - Medida catalogada: revisar politica de red del segmento
  - Referencia contextualizada: Revisar la política que permite el acceso de 192.168.1.37 a 192.168.1.193 por TCP/445 y limitarlo a necesidades operativas.
  - Estado auditable: `approved_by_author`

## SQL_injection

Caso: `case-mitlive-11-65a56c45` · Origen: `edge_iiotset`

- [x] `case-mitlive-11-65a56c45::mitigation::00`

  - Medida catalogada: bloquear el payload y el origen observados mediante WAF o reglas equivalentes
  - Referencia contextualizada: Aplicar la regla al GET de 192.168.0.170 hacia 192.168.0.128:80 que contiene SLEEP(5) y lógica condicional en el parámetro id.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-11-65a56c45::mitigation::01`

  - Medida catalogada: retirar temporalmente el endpoint vulnerable si no puede protegerse de inmediato
  - Referencia contextualizada: Retirar temporalmente /DVWA/vulnerabilities/sqli_blind/ solo si no puede filtrarse de inmediato el patrón observado.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-11-65a56c45::mitigation::02`

  - Medida catalogada: revisar logs, eliminar persistencia y rotar credenciales de base de datos expuestas
  - Referencia contextualizada: Revisar los registros del endpoint por resultados del GET; eliminar persistencia o rotar credenciales solo ante evidencia adicional.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-11-65a56c45::mitigation::03`

  - Medida catalogada: usar consultas parametrizadas y evitar la concatenacion de entradas en SQL
  - Referencia contextualizada: Parametrizar la consulta que procesa id en el endpoint observado y evitar concatenar el valor recibido.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-11-65a56c45::mitigation::04`

  - Medida catalogada: validar entradas y actualizar la aplicacion y sus dependencias
  - Referencia contextualizada: Validar el parámetro id y comprobar actualizaciones de DVWA y sus dependencias en 192.168.0.128.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-11-65a56c45::mitigation::05`

  - Medida catalogada: capturar ventana de flujo para evidencia
  - Referencia contextualizada: Conservar la ventana del flujo TCP de 383 bytes que contiene el GET, la URI codificada y sus extremos.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-11-65a56c45::mitigation::06`

  - Medida catalogada: revisar politica de red del segmento
  - Referencia contextualizada: Revisar si la política permite que 192.168.0.170 acceda por HTTP a ese endpoint de 192.168.0.128:80.
  - Estado auditable: `approved_by_author`

## Uploading

Caso: `case-mitlive-12-cf413724` · Origen: `edge_iiotset`

- [x] `case-mitlive-12-cf413724::mitigation::00`

  - Medida catalogada: deshabilitar temporalmente la subida vulnerable y poner en cuarentena los ficheros recibidos
  - Referencia contextualizada: Condicionar la deshabilitación y cuarentena a confirmar que el SYN a 192.168.0.170:3333 corresponde a una subida, pues no se observa ningún fichero.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-12-cf413724::mitigation::01`

  - Medida catalogada: bloquear el origen y los indicadores asociados con la carga maliciosa
  - Referencia contextualizada: Bloquear 192.168.0.128 hacia 192.168.0.170:3333 solo si se valida su relación con una carga maliciosa; el registro solo muestra un SYN.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-12-cf413724::mitigation::02`

  - Medida catalogada: eliminar los artefactos subidos y buscar ejecucion o persistencia derivada
  - Referencia contextualizada: Buscar o eliminar artefactos únicamente si el análisis posterior confirma una subida; el evento no aporta datos de ficheros, ejecución ni persistencia.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-12-cf413724::mitigation::03`

  - Medida catalogada: aplicar listas permitidas de tipo, tamano y contenido y analizar cada fichero
  - Referencia contextualizada: Aplicar controles de tipo, tamaño y contenido si el puerto 3333 se confirma como servicio de subida; el SYN no identifica ningún fichero.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-12-cf413724::mitigation::04`

  - Medida catalogada: almacenar las cargas fuera de rutas ejecutables y actualizar el componente de subida
  - Referencia contextualizada: Revisar el almacenamiento y la actualización del componente solo si se vincula la conexión al puerto 3333 con una función de subida.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-12-cf413724::mitigation::05`

  - Medida catalogada: capturar muestra de paquetes
  - Referencia contextualizada: Conservar el SYN TCP de 192.168.0.128:54248 a 192.168.0.170:3333, con sus flags y opciones.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-12-cf413724::mitigation::06`

  - Medida catalogada: inspeccionar anomalias de protocolo
  - Referencia contextualizada: Comprobar si el SYN sin ACK, FIN ni RST y sus opciones TCP son coherentes con el servicio esperado en el puerto 3333.
  - Estado auditable: `approved_by_author`

## Vulnerability_scanner

Caso: `case-mitlive-13-ec622ca5` · Origen: `edge_iiotset`

- [x] `case-mitlive-13-ec622ca5::mitigation::00`

  - Medida catalogada: bloquear o limitar el escaner si su actividad no esta autorizada
  - Referencia contextualizada: Limitar 192.168.0.170 solo si no está autorizado a enviar el GET con basepath externo a 192.168.0.128:80.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-13-ec622ca5::mitigation::01`

  - Medida catalogada: aislar el servicio expuesto cuando el escaneo vaya acompanado de intentos de explotacion
  - Referencia contextualizada: Aislar temporalmente el servicio HTTP de 192.168.0.128 solo si la validación del GET confirma riesgo de explotación; un paquete no acredita explotación.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-13-ec622ca5::mitigation::02`

  - Medida catalogada: validar los hallazgos y corregir las vulnerabilidades confirmadas
  - Referencia contextualizada: Validar el parámetro basepath del endpoint y corregir una inclusión remota solo si la prueba confirma esa vulnerabilidad.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-13-ec622ca5::mitigation::03`

  - Medida catalogada: mantener inventario y ejecutar escaneos autorizados dentro del ciclo de parcheo
  - Referencia contextualizada: Incorporar /DVWA/include/pear/IT.php y su parámetro basepath al inventario y a escaneos autorizados del ciclo de parcheo.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-13-ec622ca5::mitigation::04`

  - Medida catalogada: retirar componentes obsoletos y reducir la superficie publicada
  - Referencia contextualizada: Revisar si el endpoint o sus componentes exponen innecesariamente basepath; el evento no aporta versiones para declarar obsolescencia.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-13-ec622ca5::mitigation::05`

  - Medida catalogada: capturar muestra de paquetes
  - Referencia contextualizada: Conservar el paquete TCP de 320 bytes con el GET y basepath=http://cirt.net/rfiinc.txt? para su validación.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-13-ec622ca5::mitigation::06`

  - Medida catalogada: inspeccionar anomalias de protocolo
  - Referencia contextualizada: Comprobar que el GET sobre la conexión TCP establecida y el valor externo de basepath sean coherentes con el servicio HTTP esperado.
  - Estado auditable: `approved_by_author`

## XSS

Caso: `case-mitlive-14-d493c1dd` · Origen: `ton_iot`

- [x] `case-mitlive-14-d493c1dd::mitigation::00`

  - Medida catalogada: bloquear el payload y el origen XSS mediante WAF o reglas de aplicacion
  - Referencia contextualizada: Aplicar una regla WAF solo si se recupera y confirma un payload XSS en la conexión HTTP a 52.28.231.150:80; el evento no incluye contenido.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-14-d493c1dd::mitigation::01`

  - Medida catalogada: deshabilitar temporalmente el formulario o endpoint vulnerable
  - Referencia contextualizada: Identificar primero el formulario o endpoint asociado a la conexión HTTP; el evento no aporta URI para deshabilitar uno concreto.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-14-d493c1dd::mitigation::02`

  - Medida catalogada: eliminar contenido persistente inyectado y revocar las sesiones potencialmente afectadas
  - Referencia contextualizada: Buscar contenido persistente y revisar sesiones solo si evidencia adicional confirma inyección; el flujo no demuestra persistencia ni afectación.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-14-d493c1dd::mitigation::03`

  - Medida catalogada: aplicar codificacion contextual de salida y saneamiento de entradas
  - Referencia contextualizada: Aplicar codificación de salida y saneamiento en el endpoint que se identifique al correlacionar la conexión; el evento no aporta parámetros.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-14-d493c1dd::mitigation::04`

  - Medida catalogada: implantar Content Security Policy y actualizar los componentes web
  - Referencia contextualizada: Revisar CSP y versiones en la aplicación que se vincule con 52.28.231.150:80; el evento no identifica componentes web.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-14-d493c1dd::mitigation::05`

  - Medida catalogada: capturar ventana de flujo para evidencia
  - Referencia contextualizada: Conservar la ventana del flujo TCP de un paquete y 52 bytes entre 192.168.1.32:55032 y 52.28.231.150:80.
  - Estado auditable: `approved_by_author`

- [x] `case-mitlive-14-d493c1dd::mitigation::06`

  - Medida catalogada: revisar politica de red del segmento
  - Referencia contextualizada: Comprobar si la política del segmento autoriza esa salida HTTP de 192.168.1.32 hacia 52.28.231.150:80.
  - Estado auditable: `approved_by_author`
