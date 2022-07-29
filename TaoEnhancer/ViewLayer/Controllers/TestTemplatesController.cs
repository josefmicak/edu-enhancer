using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Rendering;
using Microsoft.EntityFrameworkCore;
using DomainModel;
using ViewLayer.Data;

namespace ViewLayer.Controllers
{
    public class TestTemplatesController : Controller
    {
        private readonly CourseContext _context;

        public TestTemplatesController(CourseContext context)
        {
            _context = context;
        }

        // GET: TestTemplates
        public async Task<IActionResult> Index()
        {
              return _context.TestTemplates != null ? 
                          View(await _context.TestTemplates.ToListAsync()) :
                          Problem("Entity set 'CourseContext.TestTemplates'  is null.");
        }

        // GET: TestTemplates/Details/5
        public async Task<IActionResult> Details(string id)
        {
            if (id == null || _context.TestTemplates == null)
            {
                return NotFound();
            }

            var testTemplate = await _context.TestTemplates
                .FirstOrDefaultAsync(m => m.TestNumberIdentifier == id);
            if (testTemplate == null)
            {
                return NotFound();
            }

            return View(testTemplate);
        }

        // GET: TestTemplates/Create
        public IActionResult Create()
        {
            return View();
        }

        // POST: TestTemplates/Create
        // To protect from overposting attacks, enable the specific properties you want to bind to.
        // For more details, see http://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Create([Bind("TestNameIdentifier,TestNumberIdentifier,Title")] TestTemplate testTemplate)
        {
            if (ModelState.IsValid)
            {
                _context.Add(testTemplate);
                await _context.SaveChangesAsync();
                return RedirectToAction(nameof(Index));
            }
            return View(testTemplate);
        }

        // GET: TestTemplates/Edit/5
        public async Task<IActionResult> Edit(string id)
        {
            if (id == null || _context.TestTemplates == null)
            {
                return NotFound();
            }

            var testTemplate = await _context.TestTemplates.FindAsync(id);
            if (testTemplate == null)
            {
                return NotFound();
            }
            return View(testTemplate);
        }

        // POST: TestTemplates/Edit/5
        // To protect from overposting attacks, enable the specific properties you want to bind to.
        // For more details, see http://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Edit(string id, [Bind("TestNameIdentifier,TestNumberIdentifier,Title")] TestTemplate testTemplate)
        {
            if (id != testTemplate.TestNumberIdentifier)
            {
                return NotFound();
            }

            if (ModelState.IsValid)
            {
                try
                {
                    _context.Update(testTemplate);
                    await _context.SaveChangesAsync();
                }
                catch (DbUpdateConcurrencyException)
                {
                    if (!TestTemplateExists(testTemplate.TestNumberIdentifier))
                    {
                        return NotFound();
                    }
                    else
                    {
                        throw;
                    }
                }
                return RedirectToAction(nameof(Index));
            }
            return View(testTemplate);
        }

        // GET: TestTemplates/Delete/5
        public async Task<IActionResult> Delete(string id)
        {
            if (id == null || _context.TestTemplates == null)
            {
                return NotFound();
            }

            var testTemplate = await _context.TestTemplates
                .FirstOrDefaultAsync(m => m.TestNumberIdentifier == id);
            if (testTemplate == null)
            {
                return NotFound();
            }

            return View(testTemplate);
        }

        // POST: TestTemplates/Delete/5
        [HttpPost, ActionName("Delete")]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> DeleteConfirmed(string id)
        {
            if (_context.TestTemplates == null)
            {
                return Problem("Entity set 'CourseContext.TestTemplates'  is null.");
            }
            var testTemplate = await _context.TestTemplates.FindAsync(id);
            if (testTemplate != null)
            {
                _context.TestTemplates.Remove(testTemplate);
            }
            
            await _context.SaveChangesAsync();
            return RedirectToAction(nameof(Index));
        }

        private bool TestTemplateExists(string id)
        {
          return (_context.TestTemplates?.Any(e => e.TestNumberIdentifier == id)).GetValueOrDefault();
        }
    }
}
