  var SEND_INTERVAL = 5000;
  var CAPTURE_INTERVAL = 50;
  var MAX_SAVED = 200

  var MouseCache = {
    saved: [],
    add: function(mouse) {
      if (this.saved.length >= MAX_SAVED)
        this.saved.shift();
      this.saved.push(mouse);
    },
    clear: function() {
      this.saved = [];
    }
  };

  var MouseTracker = {
    dom: null,
    user_agent: navigator.userAgent,
    last_mouse: {x: 0, y:0, ts: 0},

    track_document: function () {
      this.dom = document.scrollingElement;
      this.dom.onmousemove = this.handle_mouse_event.bind(this);
    },

    handle_mouse_event: function(evt) {
      this.last_mouse = this.parse_evt(evt);
      MouseCache.add(this.last_mouse);
    },

    parse_evt: function(evt, additional) {
      var parsed = {
        xy_section: Math.ceil((Math.round((evt.clientX/this.dom.scrollWidth)*1000)/10) / 10) * 10 + '-' + Math.ceil((Math.round((evt.clientY/this.dom.scrollHeight)*1000)/10) / 10) * 10,
        xy_area: Math.ceil((Math.round((evt.clientX/this.dom.scrollWidth)*1000)/10) / 5) * 5 + '-' + Math.ceil((Math.round((evt.clientY/this.dom.scrollHeight)*1000)/10) / 5) * 5,
        xy_element: Math.ceil((Math.round((evt.clientX/this.dom.scrollWidth)*1000)/10) / 2) * 2 + '-' + Math.ceil((Math.round((evt.clientY/this.dom.scrollHeight)*1000)/10) / 2) * 2,
         ts: Date.now()
        // w: this.dom.scrollWidth,
        // h: this.dom.scrollHeight,
        // url: window.location.href
      };
      for (var add in additional) if (additional.hasOwnProperty(add)) parsed[add] = additional[add];
      return parsed;
    },

    has_moved: function () {
      return (Date.now() - this.last_mouse.ts)/1000 <= SEND_INTERVAL/1000;
    }

      
  };

    function send_to_server(){
        if (!MouseTracker.has_moved()) return;
        xy_section = []
        xy_area = []
        xy_element = []
        for (i in MouseCache.saved) {
            if (xy_section[xy_section.length-1] != MouseCache.saved[i].xy_section) {
                xy_section.push(MouseCache.saved[i].xy_section)
            }
            if (xy_area[xy_area.length-1] != MouseCache.saved[i].xy_area) {
                xy_area.push(MouseCache.saved[i].xy_area)
            }
            if (xy_element[xy_area.length-1] != MouseCache.saved[i].xy_element) {
                xy_element.push(MouseCache.saved[i].xy_element)
            }
        }
        var xmlHttp = new XMLHttpRequest();
        xmlHttp.open( "GET", 'http://127.0.0.1:5000/user-intention?xy_section='+xy_section+'&xy_area='+xy_area+'&xy_element='+xy_element, false ); // false for synchronous request
        xmlHttp.send( null );
		console.log(JSON.parse(xmlHttp.responseText)['response'])
        return JSON.parse(xmlHttp.responseText);

    }
  function check_cache() {
    if (MouseCache.saved.length !== 0) {
      send_to_server();
    }
  }

  /***************** START TRACKING *****************/
  // TOFIX: pollyfill roll-out
  if (document.scrollingElement !== null || document.scrollingElement !== undefined) {
    MouseTracker.track_document();
  
    setInterval(send_to_server, SEND_INTERVAL);

    window.addEventListener('beforeunload', check_cache);
  }