const mqtt = require('mqtt')
const client  = mqtt.connect(`${process.env.ENV_VARIABLE}:1883`)

// client.on('connect', function () {
//   client.subscribe('presence', function (err) {
//     if (!err) {
//       client.publish('presence', 'Hello mqtt')
//     }
//   })
// })

client.on('emotion', function (topic, message) {
  // message is Buffer
  console.log(message)
  client.end()
})

