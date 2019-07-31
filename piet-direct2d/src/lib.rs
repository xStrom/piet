//! The Direct2D backend for the Piet 2D graphics abstraction.

mod conv;
mod d2d;
mod dwrite;

pub use d2d::{D2DDevice, D2DFactory};
pub use dwrite::DwriteFactory;

use crate::conv::{
    affine_to_matrix3x2f, color_to_colorf, convert_stroke_style, gradient_stop_to_d2d,
    rect_to_rectf, to_point2f,
};

use std::borrow::Cow;

use winapi::um::d2d1::{
    D2D1_BITMAP_INTERPOLATION_MODE_LINEAR, D2D1_BITMAP_INTERPOLATION_MODE_NEAREST_NEIGHBOR,
    D2D1_DRAW_TEXT_OPTIONS_NONE, D2D1_LINEAR_GRADIENT_BRUSH_PROPERTIES,
    D2D1_RADIAL_GRADIENT_BRUSH_PROPERTIES,
};
use winapi::um::dcommon::{D2D1_ALPHA_MODE_IGNORE, D2D1_ALPHA_MODE_PREMULTIPLIED};

use piet::kurbo::{Affine, PathEl, Point, Rect, Shape};

use piet::{
    new_error, Color, Error, ErrorKind, FillRule, Font, FontBuilder, Gradient, ImageFormat,
    InterpolationMode, RenderContext, StrokeStyle, Text, TextLayout, TextLayoutBuilder,
};

use d2d::{Bitmap, Brush, DeviceContext, PathGeometry};
use dwrite::{TextFormat, TextFormatBuilder};

pub struct D2DRenderContext<'a> {
    factory: &'a D2DFactory,
    inner_text: D2DText<'a>,
    rt: &'a mut DeviceContext,

    /// The context state stack. There is always at least one, until finishing.
    ctx_stack: Vec<CtxState>,

    err: Result<(), Error>,
}

pub struct D2DText<'a> {
    dwrite: &'a DwriteFactory,
}

pub struct D2DFont(TextFormat);

pub struct D2DFontBuilder<'a> {
    builder: TextFormatBuilder<'a>,
    name: String,
}

pub struct D2DTextLayout(dwrite::TextLayout);

pub struct D2DTextLayoutBuilder<'a> {
    builder: dwrite::TextLayoutBuilder<'a>,
}

#[derive(Default)]
struct CtxState {
    transform: Affine,

    // Note: when we start pushing both layers and axis aligned clips, this will
    // need to keep track of which is which. But for now, keep it simple.
    n_layers_pop: usize,
}

impl<'b, 'a: 'b> D2DRenderContext<'a> {
    /// Create a new Piet RenderContext for the Direct2D DeviceContext.
    ///
    /// TODO: check signature.
    pub fn new(
        factory: &'a D2DFactory,
        dwrite: &'a DwriteFactory,
        rt: &'b mut DeviceContext,
    ) -> D2DRenderContext<'b> {
        let inner_text = D2DText { dwrite };
        D2DRenderContext {
            factory,
            inner_text: inner_text,
            rt,
            ctx_stack: vec![CtxState::default()],
            err: Ok(()),
        }
    }

    fn current_transform(&self) -> Affine {
        // This is an unwrap because we protect the invariant.
        self.ctx_stack.last().unwrap().transform
    }

    fn pop_state(&mut self) {
        // This is an unwrap because we protect the invariant.
        let old_state = self.ctx_stack.pop().unwrap();
        for _ in 0..old_state.n_layers_pop {
            self.rt.pop_layer();
        }
    }
}

fn path_from_shape(
    d2d: &D2DFactory,
    is_filled: bool,
    shape: impl Shape,
    fill_rule: FillRule,
) -> Result<PathGeometry, Error> {
    let mut path = d2d.create_path_geometry()?;
    let mut sink = path.open()?;
    sink.set_fill_mode(fill_rule);
    let mut need_close = false;
    for el in shape.to_bez_path(1e-3) {
        match el {
            PathEl::MoveTo(p) => {
                if need_close {
                    sink.end_figure(false);
                }
                sink.begin_figure(to_point2f(p), is_filled);
                need_close = true;
            }
            PathEl::LineTo(p) => {
                sink.add_line(to_point2f(p));
            }
            PathEl::QuadTo(p1, p2) => {
                sink.add_quadratic_bezier(to_point2f(p1), to_point2f(p2));
            }
            PathEl::CurveTo(p1, p2, p3) => {
                sink.add_bezier(to_point2f(p1), to_point2f(p2), to_point2f(p3));
            }
            PathEl::ClosePath => {
                sink.end_figure(true);
                need_close = false;
            }
        }
    }
    if need_close {
        sink.end_figure(false);
    }
    sink.close()?;
    Ok(path)
}

impl<'a> RenderContext for D2DRenderContext<'a> {
    type Brush = Brush;

    type Text = D2DText<'a>;

    type TextLayout = D2DTextLayout;

    type Image = Bitmap;

    fn status(&mut self) -> Result<(), Error> {
        std::mem::replace(&mut self.err, Ok(()))
    }

    fn clear(&mut self, color: Color) {
        self.rt.clear(color_to_colorf(color));
    }

    fn solid_brush(&mut self, color: Color) -> Brush {
        self.rt
            .create_solid_color(color_to_colorf(color))
            .expect("error creating solid brush")
    }

    fn gradient(&mut self, gradient: Gradient) -> Result<Brush, Error> {
        match gradient {
            Gradient::Linear(linear) => {
                let props = D2D1_LINEAR_GRADIENT_BRUSH_PROPERTIES {
                    startPoint: to_point2f(linear.start),
                    endPoint: to_point2f(linear.end),
                };
                let stops: Vec<_> = linear.stops.iter().map(gradient_stop_to_d2d).collect();
                let stops = self.rt.create_gradient_stops(&stops)?;
                let result = self.rt.create_linear_gradient(&props, &stops)?;
                Ok(result)
            }
            Gradient::Radial(radial) => {
                let props = D2D1_RADIAL_GRADIENT_BRUSH_PROPERTIES {
                    center: to_point2f(radial.center),
                    gradientOriginOffset: to_point2f(radial.origin_offset),
                    radiusX: radial.radius as f32,
                    radiusY: radial.radius as f32,
                };
                let stops: Vec<_> = radial.stops.iter().map(gradient_stop_to_d2d).collect();
                let stops = self.rt.create_gradient_stops(&stops)?;
                let result = self.rt.create_radial_gradient(&props, &stops)?;
                Ok(result)
            }
        }
    }

    fn fill(&mut self, shape: impl Shape, brush: &Self::Brush, fill_rule: FillRule) {
        // TODO: various special-case shapes, for efficiency
        match path_from_shape(self.factory, true, shape, fill_rule) {
            Ok(path) => self.rt.fill_geometry(&path, brush, None),
            Err(e) => self.err = Err(e),
        }
    }

    fn stroke(
        &mut self,
        shape: impl Shape,
        brush: &Self::Brush,
        width: f64,
        style: Option<&StrokeStyle>,
    ) {
        // TODO: various special-case shapes, for efficiency
        let path = match path_from_shape(self.factory, false, shape, FillRule::EvenOdd) {
            Ok(path) => path,
            Err(e) => {
                self.err = Err(e);
                return;
            }
        };
        let width = width as f32;
        let style = if let Some(style) = style {
            Some(convert_stroke_style(self.factory, style, width).expect("TODO"))
        } else {
            None
        };
        self.rt.draw_geometry(&path, brush, width, style.as_ref());
    }

    fn clip(&mut self, shape: impl Shape, fill_rule: FillRule) {
        // TODO: set size based on bbox of shape.
        let layer = match self.rt.create_layer(None) {
            Ok(layer) => layer,
            Err(e) => {
                self.err = Err(e.into());
                return;
            }
        };
        let path = match path_from_shape(self.factory, true, shape, fill_rule) {
            Ok(path) => path,
            Err(e) => {
                self.err = Err(e);
                return;
            }
        };
        self.rt.push_layer_mask(&path, &layer);
        self.ctx_stack.last_mut().unwrap().n_layers_pop += 1;
    }

    fn text(&mut self) -> &mut Self::Text {
        &mut self.inner_text
    }

    fn draw_text(&mut self, layout: &Self::TextLayout, pos: impl Into<Point>, brush: &Self::Brush) {
        // TODO: set ENABLE_COLOR_FONT on Windows 8.1 and above, need version sniffing.
        let mut line_metrics = Vec::with_capacity(1);
        layout.0.get_line_metrics(&mut line_metrics);
        if line_metrics.is_empty() {
            // Layout is empty, don't bother drawing.
            return;
        }
        // Direct2D takes upper-left, so adjust for baseline.
        let mut pos = to_point2f(pos.into());
        pos.y -= line_metrics[0].baseline;
        let text_options = D2D1_DRAW_TEXT_OPTIONS_NONE;

        self.rt
            .draw_text_layout(pos, &layout.0, brush, text_options);
    }

    fn save(&mut self) -> Result<(), Error> {
        let new_state = CtxState {
            transform: self.current_transform(),
            n_layers_pop: 0,
        };
        self.ctx_stack.push(new_state);
        Ok(())
    }

    fn restore(&mut self) -> Result<(), Error> {
        if self.ctx_stack.len() <= 1 {
            return Err(new_error(ErrorKind::StackUnbalance));
        }
        self.pop_state();
        // Move this code into impl to avoid duplication with transform?
        self.rt
            .set_transform(&affine_to_matrix3x2f(self.current_transform()));
        Ok(())
    }

    // Discussion question: should this subsume EndDraw, with BeginDraw on
    // D2DRenderContext creation? I'm thinking not, as the shell might want
    // to do other stuff, possibly related to incremental paint.
    fn finish(&mut self) -> Result<(), Error> {
        if self.ctx_stack.len() != 1 {
            return Err(new_error(ErrorKind::StackUnbalance));
        }
        self.pop_state();
        std::mem::replace(&mut self.err, Ok(()))
    }

    fn transform(&mut self, transform: Affine) {
        self.ctx_stack.last_mut().unwrap().transform *= transform;
        self.rt
            .set_transform(&affine_to_matrix3x2f(self.current_transform()));
    }

    fn make_image(
        &mut self,
        width: usize,
        height: usize,
        buf: &[u8],
        format: ImageFormat,
    ) -> Result<Self::Image, Error> {
        // TODO: this method _really_ needs error checking, so much can go wrong...
        let alpha_mode = match format {
            ImageFormat::Rgb => D2D1_ALPHA_MODE_IGNORE,
            ImageFormat::RgbaPremul | ImageFormat::RgbaSeparate => D2D1_ALPHA_MODE_PREMULTIPLIED,
            _ => return Err(new_error(ErrorKind::NotSupported)),
        };
        let buf = match format {
            ImageFormat::Rgb => {
                let mut new_buf = vec![255; width * height * 4];
                for i in 0..width * height {
                    new_buf[i * 4 + 0] = buf[i * 3 + 0];
                    new_buf[i * 4 + 1] = buf[i * 3 + 1];
                    new_buf[i * 4 + 2] = buf[i * 3 + 2];
                }
                Cow::from(new_buf)
            }
            ImageFormat::RgbaSeparate => {
                let mut new_buf = vec![255; width * height * 4];
                // TODO (performance): this would be soooo much faster with SIMD
                fn premul(x: u8, a: u8) -> u8 {
                    let y = (x as u16) * (a as u16);
                    ((y + (y >> 8) + 0x80) >> 8) as u8
                }
                for i in 0..width * height {
                    let a = buf[i * 4 + 3];
                    new_buf[i * 4 + 0] = premul(buf[i * 4 + 0], a);
                    new_buf[i * 4 + 1] = premul(buf[i * 4 + 1], a);
                    new_buf[i * 4 + 2] = premul(buf[i * 4 + 2], a);
                    new_buf[i * 4 + 3] = a;
                }
                Cow::from(new_buf)
            }
            ImageFormat::RgbaPremul => Cow::from(buf),
            // This should be unreachable, we caught it above.
            _ => return Err(new_error(ErrorKind::NotSupported)),
        };
        let bitmap = self.rt.create_bitmap(width, height, &buf, alpha_mode)?;
        Ok(bitmap)
    }

    fn draw_image(
        &mut self,
        image: &Self::Image,
        rect: impl Into<Rect>,
        interp: InterpolationMode,
    ) {
        let interp = match interp {
            InterpolationMode::NearestNeighbor => D2D1_BITMAP_INTERPOLATION_MODE_NEAREST_NEIGHBOR,
            InterpolationMode::Bilinear => D2D1_BITMAP_INTERPOLATION_MODE_LINEAR,
        };
        self.rt
            .draw_bitmap(&image, &rect_to_rectf(rect.into()), 1.0, interp, None);
    }
}

impl<'a> D2DText<'a> {
    /// Create a new factory that satisfies the piet `Text` trait given
    /// the (platform-specific) dwrite factory.
    pub fn new(dwrite: &'a DwriteFactory) -> D2DText<'a> {
        D2DText { dwrite }
    }
}

impl<'a> Text for D2DText<'a> {
    type FontBuilder = D2DFontBuilder<'a>;
    type Font = D2DFont;
    type TextLayoutBuilder = D2DTextLayoutBuilder<'a>;
    type TextLayout = D2DTextLayout;

    fn new_font_by_name(&mut self, name: &str, size: f64) -> Result<Self::FontBuilder, Error> {
        // Note: the name is cloned here, rather than applied using `with_family` for
        // lifetime reasons. Maybe there's a better approach.
        let builder = TextFormatBuilder::new(self.dwrite).size(size as f32);
        Ok(D2DFontBuilder {
            builder,
            name: name.to_owned(),
        })
    }

    fn new_text_layout(
        &mut self,
        font: &Self::Font,
        text: &str,
    ) -> Result<Self::TextLayoutBuilder, Error> {
        Ok(D2DTextLayoutBuilder {
            builder: dwrite::TextLayoutBuilder::new(self.dwrite)
                .format(&font.0)
                .text(text),
        })
    }
}

impl<'a> FontBuilder for D2DFontBuilder<'a> {
    type Out = D2DFont;

    fn build(self) -> Result<Self::Out, Error> {
        Ok(D2DFont(self.builder.family(&self.name).build()?))
    }
}

impl Font for D2DFont {}

impl<'a> TextLayoutBuilder for D2DTextLayoutBuilder<'a> {
    type Out = D2DTextLayout;

    fn build(self) -> Result<Self::Out, Error> {
        Ok(D2DTextLayout(
            self.builder
                .width(1e6) // TODO: probably want to support wrapping
                .height(1e6)
                .build()?,
        ))
    }
}

impl TextLayout for D2DTextLayout {
    fn width(&self) -> f64 {
        self.0.get_metrics().width as f64
    }
}
