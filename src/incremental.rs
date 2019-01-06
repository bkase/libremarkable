use std::collections::HashMap;
use std::hash::Hash;

pub trait Semigroup {
    fn apply(&mut self, rhs: Self);
}

pub trait Monoid: Semigroup {
    fn one() -> Self;
}

pub struct Last<A>(Option<A>);

impl<A> Semigroup for Last<A> {
    fn apply(&mut self, rhs: Last<A>) {
        match rhs {
            Last(Some(_)) =>
                *self = rhs,
            Last(None) => ()
        }
    }
}

impl<A> Monoid for Last<A> {
    fn one() -> Self { Last(None) }
}

pub trait Patch {
    type Change : Monoid;

    // returns the dirty bit (true for dirty)
    fn patch(&mut self, m: Self::Change) -> bool;
}

pub struct Atomic<A>(A);

impl<A> Patch for Atomic<A> {
    type Change = Last<A>;
    fn patch(&mut self, m: Self::Change) {
        let Atomic(lhs) = self;
        match m {
            Last(Some(next)) => {
                *lhs = next;
                true
            },
            Last(None) => false
        }
    }
}

pub struct Jet<A: Patch> {
    position : A,
    velocity : A::Change
}

impl<A: Patch> Jet {
    pub fn constant(a: A) -> Jet<A> {
        Jet {
            position: a,
            velocity: A::Change::one()
        }
    }
}

impl<A: Patch> Jet<A> {
    pub fn map_atomic<B,F>(f: F, jet_a: Jet<Atomic<A>>) -> Jet<Atomic<B>>
        where
            F: Fn(A) -> B
    {
        let Atomic(position) = jetA.position;
        let Last(velocity) = jetA.velocity;
        Jet {
            position: Atomic(f(position)),
            velocity: Last(velocity.map(f))
        }
    }
}

// TODO: Impl eq/debug
pub struct IMap<K: Hash, V>(HashMap<K,V>);

pub enum MapChange<V: Patch> {
    Add(V),
    Remove,
    Update(V::Change)
}

pub struct MapChanges<K: Hash, V: Patch>(HashMap<K,MapChange<V>>);

impl<K: Eq + Hash, V: Patch> Semigroup for MapChanges<K, V> {
    fn apply(&mut self, MapChanges(mut rhs): MapChanges<K,V>) {
        let MapChanges(lhs) = self;
        for (k, m2) in rhs.drain() {
            match lhs.get_mut(&k) {
              Some(m1) =>
                  match m2 {
                    MapChange::Add(v) =>
                        *m1 = MapChange::Add(v),
                    MapChange::Remove =>
                        *m1 = MapChange::Remove,
                    MapChange::Update(dv2) =>
                        match m1 {
                            MapChange::Add(v) =>
                                v.patch(dv2),
                            MapChange::Remove => (),
                            MapChange::Update(dv1) =>
                                dv1.apply(dv2)
                        }
                  },
              None => ()
            }
        }
    }
}

impl<K: Eq + Hash, V: Patch> Monoid for MapChanges<K, V> {
    fn one() -> MapChanges<K,V> {
        return MapChanges(HashMap::new())
    }
}

impl<K: Eq + Hash, V: Patch> Patch for IMap<K, V> {
    type Change = MapChanges<K, V>;

    fn patch(&mut self, MapChanges(mut rhs) : MapChanges<K, V>) -> bool {
        let IMap(lhs) = self;
        let mut dirty = false;
        for (k, m2) in rhs.drain() {
            match lhs.get_mut(&k) {
                Some(v) =>
                    match m2 {
                        MapChange::Update(dv) => {
                            dirty = dirty || v.patch(dv);
                        },
                        MapChange::Remove => {
                            let _ = lhs.remove(&k);
                            dirty = true;
                        },
                        MapChange::Add(v) => {
                            let _ = lhs.insert(k, v);
                            dirty = true;
                        }
                    }
                None =>
                    match m2 {
                        MapChange::Add(v) => {
                            let _ = lhs.insert(k, v);
                            dirty = true;
                        },
                        _ => ()
                    }
            }
        }
        dirty
    }
}

pub fn insert<K: Eq + Hash, V: Patch>(k: K, v: V) -> MapChanges<K, V> {
    let mut m = HashMap::new();
    m.insert(k, MapChange::Add(v));
    MapChanges(m)
}

pub fn remove<K: Eq + Hash, V: Patch>(k: K, v: V) -> MapChanges<K, V> {
    let mut m = HashMap::new();
    m.insert(k, MapChange::Remove);
    MapChanges(m)
}

pub fn update_at<K: Eq + Hash, V: Patch>(k: K, v: V, c: V::Change) -> MapChanges<K, V> {
    let mut m = HashMap::new();
    m.insert(k, MapChange::Update(c));
    MapChanges(m)
}

// A new map where values can change but keys are fixed
pub fn static_<K: Clone + Eq + Hash, V: Patch>(mut xs: HashMap<K, Jet<V>>) -> Jet<IMap<K,V>> {

    let (position, velocity) : (HashMap<K, V>, HashMap<K, MapChange<V>>) =
        xs.drain()
            .map(|(k, jet_v)|
                 (
                     (k.clone(), jet_v.position),
                     (k, MapChange::Update(jet_v.velocity))
                 )
            )
            .unzip();
    Jet {
        position: IMap(position)
    ,   velocity: MapChanges(velocity)
    }
}

// A new map from a single k-v pair
pub fn singleton<K: Clone + Eq + Hash, V: Patch>(k: K, v: Jet<V>) -> Jet<IMap<K,V>> {
    let mut m = HashMap::new();
    m.insert(k, v);
    static_(m)
}

use framebuffer::cgmath;
use framebuffer::common;
use framebuffer::common::{color, mxcfb_rect};
use framebuffer::refresh::PartialRefreshMode;
use framebuffer::FramebufferDraw;
use framebuffer::FramebufferRefresh;

pub struct TextView {
    position: Atomic<cgmath::Point2>,
    last_drawn_rect: Option<common::mxcfb_rect>,
    text: Atomic<String>
}

pub struct TextViewChanges {
    position: Last<cgmath::Point2>,
    text: Last<String>
}

impl Semigroup for TextViewChanges {
    fn apply(&mut self, rhs: TextViewChanges) {
        self.position.apply(rhs.position);
        self.text.apply(rhs.scale);
    }
}

impl Monoid for TextViewChanges {
    fn one() -> TextViewChanges {
        TextViewChanges {
            position: Last::one(),
            text: Last::one()
        }
    }
}

impl Patch for TextView {
    type Change = TextViewChanges;

    fn patch(&mut self, rhs: TextViewChanges) -> bool {
        return self.position.patch(rhs.position) ||
            self.text.patch(rhs.text);
    }
}

impl TextView {
    pub fn patch_and_redraw(&mut self,
                            app: &mut appctx::ApplicationContext,
                            patch: TextViweChanges) {
        let dirty = self.patch(patch);
        if !dirty {
           return;
        }

        let framebuffer = app.get_framebuffer_ref();

        let old_filled_rect = match self.last_drawn_rect {
            Some(rect) => {
                // Clear the background on the last occupied region
                framebuffer.fill_rect(
                    rect.top_left().cast().unwrap(),
                    rect.size(),
                    color::WHITE
                );

                // We have filled the old_filled_rect, now we need to also refresh that but if
                // only if it isn't at the same spot. Otherwise we will be refreshing it for no
                // reason and showing a blank frame. There is of course still a caveat since we don't
                // know the dimensions of a drawn text before it is actually drawn.
                // TODO: Take care of the point above ^
                if rect.top_left() != self.position.cast().unwrap() {
                    framebuffer.partial_refresh(
                        &rect,
                        PartialRefreshMode::Wait,
                        common::waveform_mode::WAVEFORM_MODE_DU,
                        common::display_temp::TEMP_USE_REMARKABLE_DRAW,
                        common::dither_mode::EPDC_FLAG_USE_DITHERING_PASSTHROUGH,
                        0,
                        false,
                    );
                }

                rect
            }
            None => mxcfb_rect::invalid(),
        };

        let rect = app.display_text(
            self.position.cast().unwrap(),
            color::WHITE,
            35.0,
            2,
            8,
            text.to_string(),
            UIConstraintRefresh::Refresh
        );

        if let Some(last_rect) = self.last_drawn_rect {
            if last_rect != rect {
                framebuffer.partial_refresh(
                    &last_rect,
                    PartialRefreshMode::Async,
                    common::waveform_mode::WAVEFORM_MODE_DU,
                    common::display_temp::TEMP_USE_REMARKABLE_DRAW,
                    common::dither_mode::EPDC_FLAG_USE_DITHERING_PASSTHROUGH,
                    0,
                    false,
                );
            }
        }

        // We need to wait until now because we don't know the size of the active region before we
        // actually go ahead and draw it.
        self.last_drawn_rect = Some(rect);
    }
}

pub fn view_(
    position: Jet<Atomic<cgmath::Point2>>,
    text: Jet<Atomic<String>>,
) -> Jet<TextView> {
    Jet {
        position: TextView {
            position: position.position,
            last_drawn_rect: Option::default(),
            text: text.position
        },
        velocity: TextViewChanges {
            position: position.velocity,
            text: text.velocity
        }
    }
}

pub type Component<Model: Patch, F: Fn(Model::Change) -> ()> =
    fn(Jet<Atomic<F>>, Jet<Rc<Model>>, Jet<View>);

use std::rc::Rc;
use std::cell::RefCell;

pub fn run<Model: Patch, F: Fn(Model::Change) -> ()>(
    component: Component<Model, F>,
    initial_model: Model
) {
    let initial_model_rc = Rc(initial_model);

    let shared_model = Rc(RefCell(initial_model));
    let shared_view  = Rc(RefCell(Option::default()));

    let on_change_model = Rc::clone(&shared_model);
    let on_change_view = Rc::clone(&shared_view);
    let on_change =
        | model_change | {
             let dv =
                component(Jet::constant(on_change), Jet { position: on_change_model, velocity: model_change }).velocity;
             on_change_view.borrow_mut().unwrap().patch_and_redraw(dv);
        };

    // bootstrap it
    let initial_view =
        component(Jet::constant(on_change), Jet::constant(Rc::clone(&initial_model_rc)).position;

    let mut ref v = shared_view.borrow_mut();
    *v = Some(initial_view);
    on_change(Model::Change::one());
}
