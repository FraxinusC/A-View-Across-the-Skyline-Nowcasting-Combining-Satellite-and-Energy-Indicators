/*******************************
 * 1.  Utilities
 *******************************/

/**
 * Cloud mask for Landsat 5/7 Collection 2 SR.
 *  - Uses QA_PIXEL bits: 3 = cloud shadow, 5 = snow, 7 = cloud.
 */
function maskL57sr(image) {
  var qa   = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3)    // cloud shadow
               .or(qa.bitwiseAnd(1 << 5))  // snow
               .or(qa.bitwiseAnd(1 << 7))  // cloud
               .not();
  // Keep original data-quality mask as well
  var dataMask = image.mask().reduce(ee.Reducer.min());
  return image.updateMask(mask).updateMask(dataMask);
}

/**
 * Summer (May–Aug) median composite for a single year.
 * Returns an *Image* with 7 renamed bands.
 */
function summerComposite(year) {
  var y        = ee.Number(year).int();
  var comp     = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
                   .filterDate(ee.Date.fromYMD(y, 5, 1),
                               ee.Date.fromYMD(y, 8, 30))
                   .map(maskL57sr)
                   .median();

  // Rename bands to keep them unique after .toBands()
  var inBands  = ['SR_B3', 'SR_B2', 'SR_B1', 'SR_B4', 'SR_B5', 'ST_B6', 'SR_B7'];
  var outBands = ['red_', 'green_', 'blue_', 'B4_', 'B5_', 'B6_', 'B7_']
                   .map(function(prefix){ return ee.String(prefix).cat(y.format()); });
  return comp.select(inBands, outBands);
}

/*******************************
 * 2.  Build the “all-bands” stack
 *******************************/

// per-year composites (2001 – 2018)
var composites = ee.List.sequence(2001, 2018).map(summerComposite);

// land-cover reference layers
var nlcd01 = ee.Image('USGS/NLCD/NLCD2001').select('landcover').rename('lc_2001');
var nlcd11 = ee.Image('USGS/NLCD/NLCD2011').select('landcover').rename('lc_2011');
var nlcd16 = ee.Image('USGS/NLCD/NLCD2016').select('landcover').rename('lc_2016');

// stack everything into one big multi-band image
var allBands = ee.Image.pixelLonLat()          // lat / lon reference
                 .addBands([nlcd01, nlcd11, nlcd16])
                 .addBands(ee.ImageCollection(composites).toBands())   // 7 × 18 = 126 bands
                 .toFloat();

/*******************************
 * 3.  Batch export helper
 *******************************/

/**
 * Export a slice of the FeatureCollection to Drive.
 *  @param {ee.FeatureCollection} fc    – input polygons
 *  @param {Number} start               – inclusive start index
 *  @param {Number} end                 – exclusive end index
 */
function exportBatch(fc, start, end) {

  // Get server-side sub-list of features to export
  var slice = fc.toList(end).slice(start, end);

  // Convert the *indexes* to client side; we still fetch features one by one
  ee.List.sequence(0, ee.Number(end).subtract(start).subtract(1))
    .getInfo()                         // tiny list (≤25) – safe to bring client side
    .forEach(function(localIdx) {

      var feat      = ee.Feature(slice.get(localIdx));
      var geom      = feat.geometry();
      var desc      = ee.String(feat.get('STATEFP'))
                       .cat(feat.get('COUNTYFP'));          // e.g. "37183"

      Export.image.toDrive({
        image:        allBands,
        description:  desc.getInfo(),                       // needs a client-side string
        region:       geom,
        scale:        30,
        maxPixels:    7e7,
        folder:       'County_50',
        fileFormat:   'TFRecord',
        formatOptions:{
          patchDimensions: [120, 120],
          kernelSize:      [20, 20]
        }
      });

    });
}

/*******************************
 * 4.  Run the batches you need
 *      (un-comment one at a time)
 *******************************/

// var blobs = ee.FeatureCollection(shape);   // ← your county polygons

// First 25 counties
// exportBatch(blobs, 0, 25);

// Second 25
// exportBatch(blobs, 25, 50);

// Third batch (e.g. only 10 counties here)
// exportBatch(blobs, 50, 60);

// …and so on
