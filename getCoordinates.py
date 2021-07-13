from snappy import ProductIO
import snappy
import jpy

def LatLon_from_XY(product, x, y):
    geoPosType = jpy.get_type('org.esa.snap.core.datamodel.GeoPos')
    geocoding = product.getSceneGeoCoding()
    geo_pos = geocoding.getGeoPos(snappy.PixelPos(x, y), geoPosType())
    if str(geo_pos.lat)=='nan':
        raise ValueError('x, y pixel coordinates not in this product')
    else:
        return geo_pos.lat, geo_pos.lon

product = ProductIO.readProduct('ShipTest/S1A_IW_GRDH_1SDV_20200718T171436_20200718T171501_033512_03E220_DEFC_msk.tif')
print(LatLon_from_XY(product,0,0))